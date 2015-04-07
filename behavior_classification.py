# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 21:27:23 2015

@author: mr243268
"""

import numpy as np
from loader import load_dynacomp, load_msdl_names_and_coords,\
                   load_dynacomp_fc
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.learning_curve import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
import matplotlib.pyplot as plt


def classification_learning_curves(X, y, title=''):
    """ Computes and plots learning curves of regression models of X and y
    """
    
    # Ridge classification
    rdgc = RidgeClassifierCV(alphas=np.logspace(-3, 3, 7))

    # Support Vector classification    
    svc = SVC()
    
    # Linear Discriminant Analysis
    lda = LDA()
    
    # Logistic Regression
    logit = LogisticRegression(penalty='l2', random_state=42)

    estimator_str = ['svc', 'lda', 'rdgc', 'logit']

    # train size
    train_size = np.linspace(.2, .9, 8)    
    
    # Compute learning curves
    for e in estimator_str:
        estimator = eval(e)
        ts, _, scores = learning_curve(estimator, X, y,
                                       train_sizes=train_size, cv=8)
        plt.plot(train_size, np.mean(scores, axis=1))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(estimator_str, loc='best')
    plt.xlabel('Train size', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    ymin,ymax = plt.ylim()
    plt.ylim(ymin, ymax + .1)
    plt.grid()
    plt.title('Classification ' + title, fontsize=16)


def pairwise_classification(X, y, title=''):
    """ Computes and plots accuracy of pairwise classification model
    """
    
    # Ridge classification
    rdgc = RidgeClassifierCV(alphas=np.logspace(-3, 3, 7))

    # Support Vector classification    
    svc = SVC()
    
    # Linear Discriminant Analysis
    lda = LDA()
    
    # Logistic Regression
    logit = LogisticRegression(penalty='l2', random_state=42)

    estimator_str = ['svc', 'lda', 'rdgc', 'logit']

    # train size
    train_size = np.linspace(.2, .9, 8)

    for e in estimator_str:
        estimator = eval(e)
        mean_acc = []
        for ts in train_size:
            sss = StratifiedShuffleSplit(y, n_iter=50, train_size=ts, 
                                         random_state=42)
            # Compute accuracies
            accuracy = []
            for train, test in sss:
                estimator.fit(X[train], y[train])
                accuracy.append(estimator.score(X[test], y[test]))
            mean_acc.append(np.mean(accuracy))
        plt.plot(train_size, mean_acc)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(estimator_str, loc='best')
    plt.xlabel('Train size', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    ymin,ymax = plt.ylim()
    plt.ylim(ymin, ymax + .1)
    plt.grid()
    plt.title('Classification ' + title, fontsize=16)
            
        

##############################################################################
# Load data
session = 'func1'
msdl = True
dataset = load_dynacomp()

# Roi names
roi_names = sorted(dataset.rois[0].keys())
msdl_str = ''
if msdl:
    roi_names, roi_coords = load_msdl_names_and_coords()
    msdl_str='msdl'

# Take only the lower diagonal values
ind = np.tril_indices(len(roi_names), k=-1)

# Label vector y
groups = ['avn', 'v', 'av']
y = np.zeros(len(dataset.subjects))
yn = np.zeros(len(dataset.subjects))
for i, group in enumerate(['v', 'av']):
    y[dataset.group_indices[group]] = i + 1
    yn[dataset.group_indices[group]] = 1

# Do classification for each metric
for metric in ['pc', 'gl', 'gsc']:
    
    # 3 groups classification
    X = []        
    for i, subject_id in enumerate(dataset.subjects):
        X.append(load_dynacomp_fc(subject_id, session=session,
                                  metric=metric, msdl=msdl)[ind])
    X = np.array(X)
    plt.figure()
    classification_learning_curves(X, y, title='_'.join([metric,
                                                         session, msdl_str]))
    
    # pairwise classification
    for i in range(2):
        for j in range(i+1, 3):
            gr_i = dataset.group_indices[groups[i]]
            gr_j = dataset.group_indices[groups[j]]
            Xp = np.vstack((X[gr_i, :], X[gr_j, :]))
            yp = np.array([0] * len(gr_i) + [1] * len(gr_j))
            plt.figure()
            pairwise_classification(Xp, yp, title='_'.join([groups[i],
                                                            groups[j],
                                                            metric,
                                                            session,
                                                            msdl_str]))
                                                             
    # (v + av) vs avn classification
    plt.figure()
    pairwise_classification(X, yn, title='_'.join(['(v+av)/avn',
                                                    metric,
                                                    session,
                                                    msdl_str]))

