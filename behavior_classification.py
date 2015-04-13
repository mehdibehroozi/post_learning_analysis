# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 21:27:23 2015

@author: mr243268
"""
import os
import numpy as np
from loader import load_dynacomp, load_msdl_names_and_coords,\
                   load_dynacomp_fc, load_roi_names_and_coords, set_figure_base_dir
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.lda import LDA
from sklearn.learning_curve import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from matplotlib import cm
from nilearn.plotting import plot_connectome


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
                                       train_sizes=train_size, cv=4)
        bl = plt.plot(train_size, np.mean(scores, axis=1))
        plt.fill_between(train_size,
                         np.mean(scores, axis=1) - np.std(scores, axis=1),
                         np.mean(scores, axis=1) + np.std(scores, axis=1),
                         facecolor=bl[0].get_c(),
                         alpha=0.1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(estimator_str, loc='best')
    plt.xlabel('Train size', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim([.3, .9])
    plt.grid()
    plt.title('Classification ' + title, fontsize=16)


def pairwise_classification(X, y, title=''):
    """ Computes and plots accuracy of pairwise classification model
    """
    
    # Ridge classification
    rdgc = RidgeClassifierCV(alphas=np.logspace(-3, 3, 7))

    # Support Vector classification    
    svc = LinearSVC(penalty='l1', dual=False)
    
    # Linear Discriminant Analysis
    lda = LDA()
    
    # Logistic Regression
    logit = LogisticRegression(penalty='l1', random_state=42)

    estimator_str = ['svc', 'lda', 'rdgc', 'logit']

    # train size
    train_size = np.linspace(.2, .9, 8)

    best_w = []
    best_acc = 0
    for e in estimator_str:
        estimator = eval(e)
        mean_acc = []
        std_acc = []
        for ts in train_size:
            sss = StratifiedShuffleSplit(y, n_iter=50, train_size=ts, 
                                         random_state=42)
            # Compute accuracies
            accuracy = []
            w = []
            for train, test in sss:
                estimator.fit(X[train], y[train])
                accuracy.append(estimator.score(X[test], y[test]))
                if e != 'rdgc' and e != 'lda':
                    w.append(estimator.coef_)
            acc = np.mean(accuracy)
            acc_std = np.std(accuracy)/2
            mean_acc.append(acc)
            std_acc.append(acc_std)
            if len(w) > 0 and acc > best_acc :
                best_acc = acc
                best_w = np.mean(w, axis=0)
        bl = plt.plot(train_size, mean_acc)
        plt.fill_between(train_size,
                         np.sum([mean_acc, std_acc], axis=0),
                         np.subtract(mean_acc, std_acc),
                         facecolor=bl[0].get_c(),
                         alpha=0.1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(estimator_str, loc='best')
    plt.xlabel('Train size', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim([.3, .9])
    plt.grid()
    plt.title('Classification ' + title, fontsize=16)


    if msdl:
        msdl_str = 'msdl'
    else:
        msdl_str = 'rois'
    output_folder = os.path.join(set_figure_base_dir('classification'),
                                 metric, session, msdl_str)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder, 'accuracy_' + title)
    plt.savefig(output_file)

    return best_w, best_acc
            
        

##############################################################################
# Load data
preprocs = []

preprocs.append({'preprocessing_folder': 'pipeline_2',
                 'prefix': 'resampled_wr'})
preprocs.append({'preprocessing_folder': 'pipeline_1',
                 'prefix': 'swr'})
for pr in preprocs:
    preprocessing_folder = pr['preprocessing_folder']
    prefix = pr['prefix']
    dataset = load_dynacomp(preprocessing_folder, prefix)
    for session in ['avg', 'func1', 'func2']:
        for msdl in [False, True]:
    
            print preprocessing_folder, prefix, session, msdl
            
            # Roi names and coords
            if msdl:
                roi_names, roi_coords = load_msdl_names_and_coords()
                msdl_str='msdl'
            else:
                roi_names, roi_coords  = load_roi_names_and_coords(dataset.subjects[0])
                msdl_str = ''
            
            # Take only the lower diagonal values
            ind = np.tril_indices(len(roi_names), k=-1)
            
            # Label vector y
            groups = ['avn', 'v', 'av']
            y = np.zeros(len(dataset.subjects))
            yn = np.zeros(len(dataset.subjects))
            yv = np.ones(len(dataset.subjects))
            yv[dataset.group_indices['v']] = 0
            for i, group in enumerate(['v', 'av']):
                y[dataset.group_indices[group]] = i + 1
                yn[dataset.group_indices[group]] = 1
            
            # Do classification for each metric
            for metric in ['gl','gsc', 'pc']:
                
                # 3 groups classification
                X = []
                for i, subject_id in enumerate(dataset.subjects):
                    X.append(load_dynacomp_fc(subject_id, session=session,
                                              metric=metric, msdl=msdl,
                                              preprocessing_folder=preprocessing_folder)[ind])
                X = np.array(X)
            #    plt.figure()
            #    classification_learning_curves(X, y, title='_'.join([metric,
            #                                                         session, msdl_str]))
                
                # pairwise classification
                for i in range(2):
                    for j in range(i+1, 3):
                        gr_i = dataset.group_indices[groups[i]]
                        gr_j = dataset.group_indices[groups[j]]
                        Xp = np.vstack((X[gr_i, :], X[gr_j, :]))
                        yp = np.array([0] * len(gr_i) + [1] * len(gr_j))
                        output = '_'.join([groups[i], groups[j], metric, session,
                                           msdl_str, preprocessing_folder])
                        plt.figure()
                        w,a = pairwise_classification(Xp, yp, title=output)
                        print groups[i], groups[j], a
                        t = np.zeros((len(roi_names), len(roi_names)))
                        t[ind] = np.abs(w)
                        t = (t + t.T) / 2.
                        if msdl:
                            msdl_str = 'msdl'
                        else:
                            msdl_str = 'rois'
                        output_folder = os.path.join(set_figure_base_dir('classification'),
                                                     metric, session, msdl_str)
                        if not os.path.isdir(output_folder):
                            os.makedirs(output_folder)
                        output_file = os.path.join(output_folder, 'connectome_' + output)
                        plot_connectome(t, roi_coords, title=output,
                                        output_file=output_file,
                                        annotate=True,
                                        edge_threshold='0%')
                        
            
                # 1 vs rest
                for i in range(3):
                    gr_i = dataset.group_indices[groups[i]]
                    yr = np.zeros(X.shape[0])
                    yr[gr_i] = 1
                    output = '_'.join([groups[i] + '_rest', metric, session, msdl_str,
                                       preprocessing_folder])
                    plt.figure()
                    w, a = pairwise_classification(X, yr, title=output)
            
                    print groups[i] + '_rest', a
                    if np.sum(w) == 0:
                        w[0] = 1
                    t = np.zeros((len(roi_names), len(roi_names)))
                    t[ind] = np.abs(w)
                    t = (t + t.T) / 2.
            
                    if msdl:
                        msdl_str = 'msdl'
                    else:
                        msdl_str = 'rois'
                    output_folder = os.path.join(set_figure_base_dir('classification'),
                                                 metric, session, msdl_str)
                    if not os.path.isdir(output_folder):
                        os.makedirs(output_folder)
                    output_file = os.path.join(output_folder, 'connectome_' + output)
                    plot_connectome(t, roi_coords, title=output, output_file=output_file,
                                    edge_threshold='0%')
