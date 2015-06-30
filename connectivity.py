# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 10:49:48 2015

@author: mr243268
"""

import os
import numpy as np
from scipy import stats, linalg
from sklearn.covariance import GraphLassoCV, LedoitWolf, OAS, \
                               ShrunkCovariance
from nilearn.group_sparse_covariance import GroupSparseCovarianceCV
from sklearn.datasets.base import Bunch
from sklearn.base import BaseEstimator, TransformerMixin
from nilearn.input_data import NiftiMapsMasker, NiftiSpheresMasker
import nibabel as nib
from joblib import Parallel, delayed
from embedding import CovEmbedding, vec_to_sym


def fisher_transform(corr):
    """ Returns fisher transform of correlations
    """
    return np.arctanh(corr)

def nii_shape(img):
    """ Returns the img shape
    """
    if isinstance(img, nib.Nifti1Image):
        return img.shape
    else:
        return nib.load(img).shape

def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients 
    between pairs of variables in C, controlling 
    for the remaining variables in C.


    Parameters
    ----------
    C : array-like, shape (n, p)
        Array with the different variables.
        Each column of C is taken as a variable


    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation
        of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]
 
            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)
            
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
        
    return P_corr


def do_mask_img(func, roi, masker, use_coordinates):
    if use_coordinates:
        masker.seeds = roi
        masker.radius = 5
    else:
        masker.maps_img = roi
    return masker.fit().transform(func)


def compute_connectivity_subject(conn, s):
    """ Returns connectivity of one fMRI for a given atlas
    """
    
    if conn == 'gl':
        fc = GraphLassoCV(max_iter=1000)
    elif conn == 'lw':
        fc = LedoitWolf()
    elif conn == 'oas':
        fc = OAS()
    elif conn == 'scov':
        fc = ShrunkCovariance()
    
	fc = Bunch(covariance_=0, precision_=0)
    fc.fit(s)
    ind = np.tril_indices(s.shape[1], k=-1)
    return fc.covariance_[ind], fc.precision_[ind]


class Connectivity(BaseEstimator, TransformerMixin):
    """ Connectivity Estimator
    computes the functional connectivity of a list of 4D niimgs,
    according to ROIs defined on an atlas.
    First, the timeseries on ROIs are extracted.
    Then, the connectivity is computed for each pair of ROIs.
    The result is a ravel of half symmetric matrix.
    
    Parameters
    ----------
    atlas : atlas filepath
    metric : metric name (gl, lw, oas, scov, corr, pcorr)
    mask : mask filepath
    detrend : masker param
    low_pass: masker param
    high_pass : masker param
    t_r : masker param
    smoothing : masker param
    resampling_target : masker param
    memory : masker param
    memory_level : masker param
    n_jobs : masker param
    
    Attributes
    ----------
    fc_ : functional connectivity (covariance and precision)
    """
    
    def __init__(self, metric, mask, use_coordinates=False,
                 detrend=True, low_pass=.1, high_pass=.01, t_r=3.,
                 resampling_target='data', smoothing_fwhm=6.,
                 memory='', memory_level=2, n_jobs=1):
        self.metric = metric
        self.mask = mask
        self.n_jobs = n_jobs
        self.use_coordinates = use_coordinates
        if self.use_coordinates:
            self.masker = NiftiSpheresMasker(seeds=None, radius=None,
                                             mask_img=self.mask,
                                             detrend=detrend,
                                             low_pass=low_pass,
                                             high_pass=high_pass,
                                             t_r=t_r,
                                             resampling_target=resampling_target,
                                             smoothing_fwhm=smoothing_fwhm,
                                             memory=memory,
                                             memory_level=memory_level)
        else:
            self.masker  = NiftiMapsMasker(maps_img=None,
                                           mask_img=self.mask,
                                           detrend=detrend,
                                           low_pass=low_pass,
                                           high_pass=high_pass,
                                           t_r=t_r,
                                           resampling_target=resampling_target,
                                           smoothing_fwhm=smoothing_fwhm,
                                           memory=memory,
                                           memory_level=memory_level)

    def fit(self, imgs, rois):
        """ compute connectivities
        """

        ts = Parallel(n_jobs=self.n_jobs, verbose=5)(delayed(
                     do_mask_img)(func, roi, self.masker,
                     self.use_coordinates)
                     for func, roi in zip(imgs, rois))

        if self.metric == 'correlation' or \
           self.metric == 'partial correlation' or \
           self.metric == 'tangent' :
           cov_embedding = CovEmbedding( kind=self.metric )
           p = np.asarray(vec_to_sym(cov_embedding.fit_transform(ts)))
           ind = np.tril_indices(p.shape[1], k=-1)
           self.fc_ = np.asarray([p[i, ...][ind] for i in range(p.shape[0])])
        elif self.metric == 'gsc':
            gsc = GroupSparseCovarianceCV(verbose=2)
            gsc.fit(ts)
            ind = np.tril_indices(ts[0].shape[1], k=-1)
            self.fc_ = np.array([cv[ind] for cv in gsc.covariances_.T])
            #gsc.covariances_[ind, :]
        else:
           p = Parallel(n_jobs=self.n_jobs, verbose=5)(delayed(
           compute_connectivity_subject)(self.metric, s) for s in ts)
           self.fc_ = np.asarray(p)[:, 0, :]

        return self.fc_