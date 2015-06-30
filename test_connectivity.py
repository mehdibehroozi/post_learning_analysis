# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:11:42 2015

@author: mr243268
"""
import numpy as np
from loader import load_roi_names_and_coords, load_dynacomp, dict_to_list
from connectivity import Connectivity, fisher_transform
from statsmodels.sandbox.stats.multicomp import multipletests


CACHE_DIR = '/home/mr243268/data/tmp'
dataset = load_dynacomp(preprocessing_folder='pipeline_2',
                        prefix='resampled_wr')

conn = Connectivity(metric='tangent', mask=dataset.mask,
                    memory=CACHE_DIR, n_jobs=3)

rois = map(dict_to_list, dataset.rois)
fc = conn.fit(dataset.func1, rois)


#############################
#from nilearn.datasets import fetch_nyu_rest
#nyu_func = fetch_nyu_rest()['func']
#nyu_conn = Connectivity(metric='tangent', mask=dataset.mask,
#                        t_r=2., memory=CACHE_DIR, n_jobs=3)
#nyu_fc = nyu_conn.fit(nyu_func, 25 * [ rois[0] ])

##############################

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.datasets.base import Bunch

lr = LogisticRegression()
groups = [ ['av', 'v'], ['av', 'avn'], ['v', 'avn'] ]

def classify_group(group):
    """Classification for a pair of groups
    """
    ind = np.hstack((dataset.group_indices[group[0]],
                    dataset.group_indices[group[1]]))
    #X =  fc[ind, :]
    X = StandardScaler().fit_transform(fc[ind, :])
    y = np.array([1]* len(dataset.group_indices[group[0]]) +
                 [-1]* len(dataset.group_indices[group[1]]))
    sss = StratifiedShuffleSplit(y, n_iter=50, test_size=.25, random_state=42)
    
    accuracy = []; coef = []
    for train, test in sss:
        lr.fit(X[train], y[train])
        accuracy.append(lr.score(X[test], y[test]))
        coef.append(lr.coef_)
    return Bunch(accuracy=np.array(accuracy),
                 coef=np.array(coef))

def classify_nyu(group):
    """Classification for a pair of groups
    """
    ind = dataset.group_indices[group]
    X = np.vstack((fc[ind, :], nyu_fc))
    #X = StandardScaler().fit_transform(X)
    y = np.array([1]* len(dataset.group_indices[group]) +
                 [-1]* len(nyu_fc))
    sss = StratifiedShuffleSplit(y, n_iter=50, test_size=.25, random_state=42)
    
    accuracy = []; coef = []
    for train, test in sss:
        lr.fit(X[train], y[train])
        accuracy.append(lr.score(X[test], y[test]))
        coef.append(lr.coef_)
    return Bunch(accuracy=np.array(accuracy),
                 coef=np.array(coef))


from scipy.stats import ttest_ind, ttest_1samp

def ttest_group(group, threshold):
    """T-test
    """

    #n_rois = 22
    #threshold /= n_rois*(n_rois - 1)/2.

    fc_group_1 = fc[dataset.group_indices[group[0]]]
    fc_group_2 = fc[dataset.group_indices[group[1]]]
    tv, pv = ttest_ind(fc_group_1, fc_group_2)
    pv = -np.log10(pv)
    thresh_log = -np.log10(threshold)    
    ind_threshold = np.where(pv < thresh_log)
    pv[ind_threshold] = 0
    p = np.zeros((22, 22))
    ind = np.tril_indices(22, k=-1)
    p[ind] = pv
    p = (p + p.T) / 2
    return p


def ttest_onesample(group, threshold):
    """T-test
    """
    
#    n_rois = 22
#    threshold /= n_rois*(n_rois - 1)/2.

    fc_group = fc[dataset.group_indices[group]]
    
    tv, pv = ttest_1samp(fc_group, 0.)

    pv = -np.log10(pv)              
    thresh_log = -np.log10(threshold)
    #Locate unsignificant tests
    ind_threshold = np.where(pv < thresh_log)
    #and then threshold
    pv[ind_threshold] = 0
    
    p = np.zeros((22, 22))
    ind = np.tril_indices(22, k=-1)
    p[ind] = pv
    p = (p + p.T) / 2    
    return p


def ttest_onesample_coef(coefs, threshold):
    """T-test
    """
    
    tv, pv = ttest_1samp(coefs, 0.)
    pv = -np.log10(pv)              
    thresh_log = -np.log10(threshold)
    #Locate unsignificant tests
    ind_threshold = np.where(pv < thresh_log)
    #and then threshold
    pv[ind_threshold] = 0

    cc = np.mean(coefs, axis=0)
    ind_threshold = np.where(pv < threshold)
    print 'pvalues :', ind_threshold
    cc[ind_threshold] = 0

    
    p = np.zeros((22, 22))
    ind = np.tril_indices(22, k=-1)
    p[ind] = cc
    p = (p + p.T) / 2    
    return p


from joblib import Parallel, delayed

a = Parallel(n_jobs=3, verbose=5)(delayed(classify_group)(group)
                                  for group in groups)

#nyu = Parallel(n_jobs=3, verbose=5)(delayed(classify_nyu)(group)
#                                    for group in ['v', 'av', 'avn'])


tst = Parallel(n_jobs=3, verbose=5)(delayed(ttest_group)(group, .05)
                                  for group in groups)

ost = Parallel(n_jobs=3, verbose=5)(delayed(ttest_onesample)(group, .05)
                                    for group in ['v', 'av', 'avn'])

import matplotlib.pyplot as plt
from nilearn.plotting import plot_connectome

r, c = load_roi_names_and_coords(dataset.subjects[0])

gr = ['v', 'av', 'avn']
for i in range(3):
    plot_connectome(tst[i], c, title='-'.join(groups[i]) )
    plot_connectome(ost[i], c, title=gr[i] )

plt.figure()
plt.boxplot(map(lambda x: x['accuracy'], a))
plt.show()
#names=map(lambda x:'-'.join(x), groups))

#for i in range(3):
#    coef = ttest_onesample_coef(a[i]['coef'], .001)
#    plot_connectome(np.absolute(coef), c, title='-'.join(groups[i]),
#                    edge_threshold='95%')
#
