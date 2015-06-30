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
from sklearn.cross_validation import StratifiedShuffleSplit, KFold, Bootstrap
import matplotlib.pyplot as plt
from matplotlib import cm
from nilearn.plotting import plot_connectome
from sklearn.grid_search import GridSearchCV


def train_and_test(X, y):
    """ Computes shuffle split accuracies
    """

    w = []
    scores = []
    clf = LinearSVC(penalty='l1', dual=False)
#    clf = LogisticRegression(penalty='l1', random_state=42)
#    clf = SVC()
#    kf = KFold(len(y), n_folds=len(y))

#    for train, test in kf:
#        clf.fit(X[train], y[train])
#        scores.append(clf.score(X[test], y[test]))
#    print np.sum(scores)/len(y)
#    svc = SVC(kernel='rbf', random_state=42)
#    param_grid = {'C': np.linspace(1, 10, 10),
#                  'gamma': np.linspace(0, 1, 11)}
#    clf = GridSearchCV(svc, param_grid, n_jobs=1)


    sss = StratifiedShuffleSplit(y, n_iter=100, train_size=.8, 
                                 random_state=42)
    for train, test in sss:
        clf.fit(X[train], y[train])
        scores.append(clf.score(X[test], y[test]))
        w.append(clf.coef_)
    return np.array(scores), np.mean(w, axis=0)



def pairwise_classification(X, y, title=''):
    """ Computes and plots accuracy of pairwise classification model
    """
    
    # Ridge classification
    rdgc = RidgeClassifierCV(alphas=np.logspace(-3, 3, 7))

    # Support Vector classification    
    svc = LinearSVC(penalty='l1', dual=False)
    
#     Linear Discriminant Analysis
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
for pr in preprocs[:-1]:
    preprocessing_folder = pr['preprocessing_folder']
    prefix = pr['prefix']
    dataset = load_dynacomp(preprocessing_folder, prefix)
    for session in ['func1']:
        for msdl in [False]:
    
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
            for metric in ['gl','gsc']:
                
                # Fill matrix
                X = []
                for i, subject_id in enumerate(dataset.subjects):
                    X.append(load_dynacomp_fc(subject_id, session=session,
                                              metric=metric, msdl=msdl,
                                              preprocessing_folder=preprocessing_folder)[ind])
                X = np.array(X)
                
                # pairwise classification
                s = []
                print metric
                for i in range(2):
                    for j in range(i+1, 3):
                        gr_i = dataset.group_indices[groups[i]]
                        gr_j = dataset.group_indices[groups[j]]
                        Xp = np.vstack((X[gr_i, :], X[gr_j, :]))
                        yp = np.array([0] * len(gr_i) + [1] * len(gr_j))
                        a, w = train_and_test(Xp, yp)                  
                        s.append(a)
                        print groups[i] + '/' + groups[j], str(np.median(a))
                        plt.figure(figsize=(8, 8))
                        m = np.zeros((len(roi_names), len(roi_names)))
                        m[ind] = abs(w[0, :])
                        plt.imshow(m, interpolation='nearest', cmap=cm.hot)
                        plt.xticks(range(len(roi_names)), roi_names, fontsize=16, rotation=90)                        
                        plt.yticks(range(len(roi_names)), roi_names, fontsize=16)                        
                        plt.colorbar()
                        plt.title(groups[i] + '/' + groups[j] +\
                                  str(np.median(a)))
                plt.figure()
                plt.boxplot(s)
                plt.xticks(range(1, 4), ['avn/v', 'avn/av', 'v/av'])
                plt.title(' '.join([preprocessing_folder,
                                    prefix, session, metric]))
