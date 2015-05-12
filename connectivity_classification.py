# -*- coding: utf-8 -*-
"""
Created on Tue May 12 09:42:23 2015

@author: mehdi.rahim@cea.fr
"""

import loader
import numpy as np
from nilearn.datasets import fetch_nyu_rest
from nilearn.input_data import NiftiMapsMasker
from sklearn.covariance import GraphLassoCV


##############################################################################
# Dynacomp rs-fMRI
##############################################################################
dyn_dataset = loader.load_dynacomp()
roi_imgs = loader.dict_to_list(loader.load_dynacomp_rois()[0])
roi_names, roi_coords = loader.load_roi_names_and_coords(dyn_dataset.subjects[0])
ind = np.tril_indices(len(roi_names), k=-1)

dyn_fc = []
for subject in dyn_dataset.subjects:
    dyn_fc.append(loader.load_dynacomp_fc(subject_id=subject, session='func1',
                                          metric='pc', msdl=False,
                                          preprocessing_folder='pipeline_1')[ind])
dyn_fc = np.asarray(dyn_fc)


##############################################################################
# NYU rs-fMRI
##############################################################################
nyu_func = fetch_nyu_rest()['func']
masker = NiftiMapsMasker(maps_img=roi_imgs, 
                         low_pass=.1,
                         high_pass=.01,
                         t_r=2.,
                         smoothing_fwhm=6., detrend=True, standardize=False,
                         resampling_target='maps', memory_level=0,
                         verbose=5)
masker.fit()

def mask_and_covariance(f):
    x = masker.transform(f)
    return np.corrcoef(x.T)[ind]
#    gl = GraphLassoCV(verbose=2)
#    gl.fit(x)
#    return gl.covariance_[ind]

from joblib import delayed, Parallel

nyu_fc = Parallel(n_jobs=20, verbose=5)(delayed(mask_and_covariance)(f)
                                        for f in nyu_func)
nyu_fc = np.asarray(nyu_fc)


##############################################################################
# Data preparation
##############################################################################
groups = ['v', 'av', 'avn']
data = {}
for key in groups:
    inds = np.where(np.array(dyn_dataset.group) == key)
    data[key] = dyn_fc[inds]
data['nyu'] = nyu_fc


##############################################################################
# Classification
##############################################################################

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

logit = LinearSVC(penalty='l1', dual=False)

#logit = LogisticRegression(penalty='l1', random_state=42)

def train_and_test(X, y, train, test):
    logit.fit(X[train], y[train])
    return logit.score(X[test], y[test])

pairs_all = []
accuracy_all = []
for i, gr1 in enumerate(sorted(data.keys())[:-1]):
    for j, gr2 in enumerate(sorted(data.keys())[i+1:]):
        print gr1, gr2
        X = np.concatenate((data[gr1], data[gr2]))
        X = StandardScaler().fit_transform(X)
        y = np.asarray([1] * data[gr1].shape[0] + [0] * data[gr2].shape[0])
        sss = StratifiedShuffleSplit(y, n_iter=50, test_size=.25,
                                     random_state=42)
        acc = Parallel(n_jobs=10, verbose=5)(delayed(train_and_test)
                                                    (X, y, train, test)
                                                    for train, test in sss)
        accuracy_all.append(acc)
        pairs_all.append('/'.join([gr1, gr2]))

import matplotlib.pyplot as plt
plt.figure()
plt.boxplot(accuracy_all)
plt.xticks(range(1, len(pairs_all) + 1), pairs_all)