# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:38:22 2015

@author: mr243268
"""
import numpy as np
from nilearn.datasets import fetch_nyu_rest
from nilearn.input_data import NiftiMapsMasker
from loader import load_dynacomp_rois, dict_to_list
import matplotlib.pyplot as plt

rois = load_dynacomp_rois()
dataset = fetch_nyu_rest()
X = []
maps_img = dict_to_list(rois[0])
# add mask, smoothing, filtering and detrending
masker = NiftiMapsMasker(maps_img=maps_img,
                         low_pass=.1,
                         high_pass=.01,
                         t_r=2.,
                         smoothing_fwhm=6.,
                         detrend=True,
                         standardize=False,
                         resampling_target='data',
                         memory_level=0,
                         verbose=5)
# extract the signal to x
masker.fit()

c = []
c_all = []
for f in dataset.func:
    print f
    x = masker.transform(f)
    
    from sklearn import covariance
    # compute covariance
    gl = covariance.GraphLassoCV(verbose=2)
    gl.fit(x)
    cc = np.corrcoef(x.T)
    c.append(cc[-1, -3])
    c_all.append(cc)

