# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 23:51:49 2015

@author: mr243268
"""

import os
import numpy as np
from nilearn.mass_univariate import permuted_ols
from loader import load_dynacomp_fc, load_dynacomp, load_roi_names_and_coords
from loader import load_msdl_names_and_coords
from loader import set_figure_base_dir, set_data_base_dir
import matplotlib.pyplot as plt
from matplotlib import cm
from nilearn.plotting import plot_connectome
from scipy.stats import ttest_1samp, ttest_ind
from nipy.algorithms.statistics import empirical_pvalue

def fisher_transform(corr):
    """ Returns fisher transform of correlations
    """
    return np.arctanh(corr)

def mean_coords(dataset):
    """Returns mean roi coords of all the subjects
    """
    mean_coords = []
    for subject_id in dataset.subjects:
        roi_names, roi_coords = load_roi_names_and_coords(subject_id)
        mean_coords.append(roi_coords)
    mean_coords = np.mean(mean_coords, axis=0)
    return roi_names, mean_coords 

def load_connectivity(dataset, session='func1', preprocessing_folder='pipeline_1',
                      metric='pc', msdl=True):
    """Returns the connectivity of all the subjects
    """
    conn_all = []
    for i in range(len(dataset.subjects)):
        cc = load_dynacomp_fc(dataset.subjects[i],
                              session=session,
                              preprocessing_folder=preprocessing_folder,
                              metric=metric, msdl=msdl)
        fc = cc[ind]
        conn_all.append(fc)
    conn_all = np.array(conn_all)
    return conn_all

def one_sample_ttest(fc_all, metric, threshold=0.05, session='func1',
                     preprocessing_folder='pipeline_1', mcp='bonf', 
                     z_fisher=True):
    """Perform and plot one sample t-tests (one per group)
    """
    n_rois = len(roi_names)
    if mcp == 'bonf':
        threshold /= n_rois*(n_rois-1)/2.

    thresh_log = -np.log10(threshold)
    # We could impose working on z-Fisher transforms in case of Pearson's
    #correlation
#    if metric == 'pc':
#        z_fisher = True

    for group in np.unique(dataset.group):
        fc_group = fc_all[dataset.group_indices[group], :]
        if z_fisher == True:
            #Fisher transformation
            fisher_gp = .5 * np.log((1+fc_group)/(1-fc_group))
            tv, pv = ttest_1samp(fisher_gp, 0.)
        else:
            tv, pv = ttest_1samp(fc_group, 0.)
        # Convert in log-scale
        pv = -np.log10(pv)
        #Locate unsignificant tests
        ind_threshold = np.where(pv < thresh_log)
        #and then threshold
        pv[ind_threshold] = 0
        
        fc_group_mean = np.mean(fc_group, axis=0)
        if metric == 'pc':
            fc_group_mean /= np.std(fc_group, axis=0)
        fc_group_mean[ind_threshold] = 0
        t = np.zeros((n_rois, n_rois))
        t[ind] = fc_group_mean

        # plot connectivity matrix
        plt.figure(figsize=(8, 8))
        plt.imshow(t, interpolation='nearest', cmap=cm.hot)
        plt.colorbar()
        str_thresh = '%1.2f' %thresh_log
        title = group + '_' + session + '(-log p-val=' + str_thresh +')'
        if msdl:
            title += '_msdl'
        plt.title(title, fontsize=20)            
        plt.xticks(range(len(roi_names)), roi_names,
                   rotation='vertical', fontsize=16)
        plt.yticks(range(len(roi_names)), roi_names, fontsize=16)
        
        
        output_folder = os.path.join(set_figure_base_dir('stat/one_sample'),
                                     metric)
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        
        output_file = os.path.join(output_folder,
                                   '_'.join(['ttest', 'matrix',
                                             session, group,
                                             metric, preprocessing_folder]))
        if msdl:
            output_file += '_msdl'
        plt.savefig(output_file)
    
        # plot connectome
        output_file = os.path.join(output_folder,
                                   '_'.join(['ttest', 'connectome',
                                             session, group,
                                             metric, preprocessing_folder]))
        if msdl:
            output_file += '_msdl'
        
        t = (t + t.T) / 2.
        plt.figure(figsize=(10, 20), dpi=90)

        title = group + '_' + session + '(-log p-val=' + str_thresh +')'
        if msdl:
            title += '_msdl'
            
        try :
            plot_connectome(t, roi_coords, edge_threshold='85%',
                            title=title, output_file=output_file)
        except ValueError :
            print 'unable to plot connectome !'
        
        # save thresholded pvalues
        output_folder = set_data_base_dir('Dynacomp/stat')
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        _, of = os.path.split(output_file)
        np.save(os.path.join(output_folder, of), t)

def two_sample_ttest(fc_all, metric, threshold=.05, session='func1',
                     preprocessing_folder='pipeline_1',
                     mcp='unc',
                     z_fisher=False):
    """ perform and plot two samples t-tests
    """
    n_rois = len(roi_names)
    if mcp == 'bonf':
        threshold /= n_rois*(n_rois-1)/2.
    # We could impose working on z-Fisher transforms in case of Pearson's
    #correlation
#    if metric == 'pc':
#        z_fisher = True
     
    thresh_log = -np.log10(threshold)
    str_thresh = '%1.2f' %thresh_log
    groups = np.unique(dataset.group)
    for i in range(len(groups) - 1):
        for j in range(i + 1, len(groups)):
            fc_group_1 = fc_all[dataset.group_indices[groups[i]], :]
            fc_group_2 = fc_all[dataset.group_indices[groups[j]], :]     
            print 'Group comparison: %s vs %s' %(groups[i], groups[j])
            #Fisher transformation
            if z_fisher == True:
                fisher_group1 = .5 * np.log((1+fc_group_1)/(1-fc_group_1))
                fisher_group2 = .5 * np.log((1+fc_group_2)/(1-fc_group_2))
                tv, pv = ttest_ind(fisher_group1, fisher_group2)
            else:
                tv, pv = ttest_ind(fc_group_1, fc_group_2)

#            zth = empirical_pvalue.gaussian_fdr_threshold(pv, threshold)
#            above_th = pv > zth
#            # FDR-corrected p-values
#            fdr_pvalue = empirical_pvalue.gaussian_fdr(pv)[above_th]
#            print fdr_pvalue
#            ind_fdr = np.where( pv < zth )
#            fdr_p = np.zeros((n_rois, n_rois))
#            fdr_p[above_th] = fdr_pvalue
#            p = np.copy(fdr_p)

            pv = -np.log10(pv)              
            ind_threshold = np.where(pv < thresh_log)
            pv[ind_threshold] = 0           
            p = np.zeros((n_rois, n_rois))
            p[ind] = pv
    
            #plot connectivity matrix
            plt.figure(figsize=(8, 8))
            plt.imshow(p, interpolation='nearest', cmap=cm.hot)
            plt.colorbar()
            title = groups[i] + ' / ' + groups[j] + '_' + session + \
                    '(p-val <' + str_thresh +')'
                
            if msdl:
                title += '_msdl'
                
            plt.title(title, fontsize=20)
            plt.xticks(range(len(roi_names)), roi_names,
                       rotation='vertical', fontsize=16)
            plt.yticks(range(len(roi_names)), roi_names, fontsize=16)


            output_folder = os.path.join(set_figure_base_dir('stat/two_sample'),
                                         metric)
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            
            output_file = os.path.join(output_folder,
                                       '_'.join(['ttest2', 'matrix',
                                                 session, groups[i], groups[j],
                                                 metric, preprocessing_folder]))
            if msdl:
                output_file += '_msdl'
            
            plt.savefig(output_file)

            # plot connectome
            output_file = os.path.join(output_folder,
                                       '_'.join(['ttest2', 'connectome',
                                                 session, groups[i], groups[j],
                                                 metric, preprocessing_folder]))

            if msdl:
                output_file += '_msdl'

            p = (p + p.T) / 2
            plt.figure(figsize=(10, 20), dpi=90)
            title = groups[i] + ' / ' + groups[j] + '_' + \
                    session + '(p-val <' + str_thresh +')'
            if msdl:
                title += '_msdl'
            try:
                plot_connectome(p, roi_coords, edge_threshold='85%',
                                title=title,
                                output_file=output_file)
            except ValueError :
                print 'unable to plot connectome !'

def permuted_two_sample_ttest(fc_all, metric, threshold=0.05, session='func1',
                              mcp='unc', z_fisher=False):
    """ Compute and plot permutated two-sample t-tests
    """
    groups = np.unique(dataset.group)
    
    n_rois = len(roi_names)
    if mcp == 'bonf':
        threshold /= n_rois*(n_rois-1)/2.
    thresh_log = -np.log10(threshold)
    str_thresh = '%1.2f' %thresh_log

    for i in range(len(groups) - 1):
        for j in range(i + 1, len(groups)):
            fc_group_1 = fc_all[dataset.group_indices[groups[i]], :]
            fc_group_2 = fc_all[dataset.group_indices[groups[j]], :]
            
            fc_groups = np.concatenate((fc_group_1, fc_group_2))
            y = np.array(fc_group_1.shape[0]*[1] + fc_group_2.shape[0]*[0])
            y = np.array([y, 1-y]).T
    
            pvals, tvals, h0 = permuted_ols(fc_groups, y,
                                            two_sided_test=True,
                                            n_perm=10000, random_state=42, n_jobs=3)
    
            pvals = pvals[:, 0]
            ind_threshold = np.where(pvals < thresh_log)
            pvals[ind_threshold] = 0
            
            p = np.zeros((n_rois, n_rois))
            p[ind] = pvals
            plt.figure(figsize=(8, 8))
            plt.imshow(p, interpolation='nearest', cmap=cm.hot)
            plt.colorbar()
            
            title = groups[i] + ' / ' + groups[j] + '_' + session
            if msdl:
                title += '_msdl'
            plt.title(title, fontsize=20)
            plt.xticks(range(len(roi_names)), roi_names,
                       rotation='vertical', fontsize=16)
            plt.yticks(range(len(roi_names)), roi_names, fontsize=16)


            output_folder = os.path.join(set_figure_base_dir('stat/two_sample'),
                                         metric)
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            
            output_file = os.path.join(output_folder,
                                       '_'.join(['ttest2', 'perm', 'matrix',
                                                 session, groups[i], groups[j],
                                                 metric, preprocessing_folder]))
            if msdl:
                output_file += '_msdl'

            plt.savefig(output_file)
            # plot connectome

            output_file = os.path.join(output_folder,
                                       '_'.join(['ttest2', 'perm', 'connectome',
                                                 session, groups[i], groups[j],
                                                 metric, preprocessing_folder]))

            if msdl:
                output_file += '_msdl'

            # make a symmetric matrix
            p = (p + p.T) / 2
            plt.figure(figsize=(10, 20), dpi=90)
            title = groups[i] + ' / ' + groups[j] + '_' + session
            if msdl:
                title += '_msdl'
                
            try :
                plot_connectome(p, roi_coords, edge_threshold='90%',
                                title=title,
                                output_file=output_file)
            except ValueError :
                print 'unable to plot connectome !'


##############################################################################
##############################################################################
# Load data and extract only

#preprocessing_folder = 'pipeline_1'
#prefix = 'swr'
preprocessing_folder = 'pipeline_2'
prefix = 'resampled_wr'

msdl = False
#msdl = False
#Switch  between subject-dependent ROIs and MSDL atlas

dataset = load_dynacomp(preprocessing_folder=preprocessing_folder,
                        prefix=prefix)

roi_names, roi_coords = mean_coords(dataset)

# Roi names
roi_names = sorted(dataset.rois[0].keys())
if msdl:
    roi_names, roi_coords = load_msdl_names_and_coords()
nb_rois = len(roi_names)

# Lower diagonal
ind = np.tril_indices(len(roi_names), k=-1)

#session = 'func1'
#session = 'func2'
#session = 'avg'
sessions = ['func1', 'func2', 'avg']


#metric = 'pc'
#metric = 'gl' #graph-lasso
#metric = 'gsc' #group sparse covariance
metrics = ['pc', 'gl', 'gsc']

for met in metrics:
    print 'Stats analysis of FC measure: %s' %met 
    for sess in sessions:
        print 'Treating sess: %s' %sess

        fc_all = load_connectivity(dataset, sess,
                                   preprocessing_folder, met, msdl)
        if met == 'gsc':
            one_sample_ttest(fc_all, met, threshold=0.05, session=sess,
                             preprocessing_folder=preprocessing_folder,
                             mcp='unc')
            two_sample_ttest(fc_all, met, threshold=0.05, session=sess,
                             preprocessing_folder=preprocessing_folder,
                             mcp='unc')
        else:
            one_sample_ttest(fc_all, met, threshold=0.05, session=sess,
                             preprocessing_folder=preprocessing_folder)
            two_sample_ttest(fc_all, met, threshold=0.05, session=sess,
                             preprocessing_folder=preprocessing_folder)
#permuted_two_sample_ttest(metric, threshold=1.3, session=session)
