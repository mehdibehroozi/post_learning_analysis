# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 21:27:23 2015

@author: mr243268
"""

import numpy as np
from loader import load_dynacomp, load_msdl_names_and_coords,\
                   load_dynacomp_fc
from sklearn.linear_model import RidgeClassifierCV
from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt


def classification_learning_curves(X, y, metric=''):
    """ Computes and plots learning curves of regression models of X and y
    """
    
    # Ridge classification
    rdgc = RidgeClassifierCV(alphas=np.logspace(-3, 3, 7))

    # Support Vector classification    
    svc = SVC()
    
    # Linear Discriminant Analysis
    lda = LDA()
    
    # Compute learning curves
    for estimator in [svc, lda, rdgc]:
        train_size, _, scores = learning_curve(estimator, X, y,
                               train_sizes=np.linspace(.2, .95, 10), cv=8)
        plt.plot(np.linspace(.2, .95, 10), np.mean(scores, axis=1))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(['SVC', 'LDA', 'Ridge cl'], loc='best')
    plt.xlabel('Train size', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    ymin,ymax = plt.ylim()
    plt.ylim(ymin, ymax + .1)
    plt.grid()
    plt.title('Classification ' + metric, fontsize=16)


##############################################################################
# Load data
msdl = True
dataset = load_dynacomp()

# Roi names
roi_names = sorted(dataset.rois[0].keys())
if msdl:
    roi_names, roi_coords = load_msdl_names_and_coords()

# Take only the lower diagonal values
ind = np.tril_indices(len(roi_names), k=-1)

# Label vector y
groups = ['avn', 'v', 'av']
y = np.zeros(len(dataset.subjects))
for i, group in enumerate(['v', 'av']):
    y[dataset.group_indices[group]] = i + 1

# Do classification for each metric
for metric in ['pc', 'gl', 'gsc']:
    
    # 3 groups classification
    X = []        
    for i, subject_id in enumerate(dataset.subjects):
        X.append(load_dynacomp_fc(subject_id, metric=metric, msdl=msdl)[ind])
    X = np.array(X)
    plt.figure()
    classification_learning_curves(X, y, metric)