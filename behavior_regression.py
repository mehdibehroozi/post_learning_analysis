# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:42:41 2015

@author: mr243268
"""

import numpy as np
from loader import load_dynacomp, load_msdl_names_and_coords,\
                   load_dynacomp_fc
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.svm import SVR
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest


def regression_learning_curves(X, y, behav_name=''):
    """ Computes and plots learning curves of regression models of X and y
    """    

    # Feature selection
#    skb = SelectKBest()
#    X = skb.fit_transform(X, y)
    
    # Ridge Regression
    rdg = RidgeCV(alphas=np.logspace(-3, 3, 7))


    # Lasso Regression
    lasso = LassoCV(alphas=np.logspace(-3, 3, 7))
    
    # Support Vector regression
    svr = SVR(kernel='rbf', random_state=42)
    
    # Compute learning curves
    for estimator in [svr, lasso, rdg]:
        train_size, _, scores = learning_curve(estimator, X, y,
                               train_sizes=np.linspace(.2, .95, 10), cv=5)
        plt.plot(np.linspace(.2, .95, 10), np.mean(scores, axis=1))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(['SVR', 'LASSO reg', 'Ridge reg'], loc='best')
    plt.xlabel('Train size', fontsize=16)
    plt.ylabel('R^2', fontsize=16)
    ymin,ymax = plt.ylim()
    plt.ylim(ymin, ymax + .1)
    plt.grid()
    plt.title(behav_name, fontsize=16)


##############################################################################
# Load data
msdl = True
dataset = load_dynacomp()

# Behavior data
behav_data = dataset.behavior
# Add deltas
for i in range(len(behav_data)):
    for key in ['Thresh', 'RT', 'HIT_RT', 'Perf', 'Conf_mean']:
        behav_data[i]['delta' + key] = behav_data[i]['post' + key] - \
                                       behav_data[i]['pre' + key]

# Roi names
roi_names = sorted(dataset.rois[0].keys())
if msdl:
    roi_names, roi_coords = load_msdl_names_and_coords()

# Take only the lower diagonal values
ind = np.tril_indices(len(roi_names), k=-1)

# Do regression for each metric and each behav score 
for metric in ['pc', 'gl', 'gsc']:
    for key in behav_data[0].keys():
        # Construct feature matrix and output measure
        X = []
        y = []
        for i, subject_id in enumerate(dataset.subjects):
            X.append(load_dynacomp_fc(subject_id, metric=metric, msdl=msdl)[ind])
            y.append(behav_data[i][key])
        X, y = np.array(X), np.array(y)
        
        behav_name = '_'.join([key, metric])
        if msdl:
            behav_name += '_msdl'
        plt.figure()
        regression_learning_curves(X, y, behav_name)