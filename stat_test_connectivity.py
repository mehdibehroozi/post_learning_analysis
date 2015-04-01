# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 23:51:49 2015

@author: mr243268
"""

import os
import numpy as np
from nilearn.mass_univariate import permuted_ols
from loader import load_dynacomp_pc, load_dynacomp, load_roi_names_and_coords,\
                    load_dynacomp_gl, load_dynacomp_gsc
from loader import set_figure_base_dir
import matplotlib.pyplot as plt
from matplotlib import cm
from nilearn.plotting import plot_connectome
from scipy.stats import ttest_1samp, ttest_ind


def mean_coords(dataset):
    """Returns mean roi coords
    """
    mean_coords = []
    for subject_id in dataset.subjects:
        roi_names, roi_coords = load_roi_names_and_coords(subject_id)
        mean_coords.append(roi_coords)
    mean_coords = np.mean(mean_coords, axis=0)
    return roi_names, mean_coords 

def load_connectivity(dataset, session='func1', connectivity='pc'):
    """Returns the connectivity of all the subjects
    """
    conn_all = []
    for i in range(len(dataset.subjects)):
        if connectivity == 'gl':
            cc = load_dynacomp_gl(dataset.subjects[i], session=session)['covariance']
        elif connectivity == 'gsc':
            cc = load_dynacomp_gsc(dataset.subjects[i], session=session)['covariance']
        else:
            cc = load_dynacomp_pc(dataset.subjects[i], session=session)['correlation']
        
        pc = cc[ind]
        conn_all.append(pc)
    conn_all = np.array(conn_all)
    return conn_all


def one_sample_ttest():
    """Plots one sample ttest per group
    """
    n_rois = len(dataset.rois[0])    
    for group in np.unique(dataset.group):
        pc_group = pc_all[dataset.group_indices[group], :]
        tv, pv = ttest_1samp(pc_group, 0.)
        pv = -np.log10(pv)

        ind_threshold = np.where(pv < 3.67)
        pv[ind_threshold] = 0
        
        pc_group_mean = np.mean(pc_group, axis=0)
        if connectivity == 'pc':
            pc_group_mean /= np.std(pc_group, axis=0)
        pc_group_mean[ind_threshold] = 0

        t = np.zeros((n_rois, n_rois))
        t[ind] = pc_group_mean
        # plot connectivity matrix
        plt.figure(figsize=(8, 8))
        plt.imshow(t, interpolation='nearest', cmap=cm.hot)
        plt.colorbar()
        plt.title(title=group + '_' + session, fontsize=20)
        plt.xticks(range(len(roi_names)), roi_names,
                   rotation='vertical', fontsize=16)
        plt.yticks(range(len(roi_names)), roi_names, fontsize=16)
        output_file = os.path.join(set_figure_base_dir(), 'connectivity',
                                   'ttest_connectivity_' + session + '_' + group)
        plt.savefig(output_file)
    
        # plot connectome
        output_file = os.path.join(set_figure_base_dir(), 'connectivity',
                                   'ttest_connectome_' + session + '_' + group)
        t = (t + t.T) / 2
        plt.figure(figsize=(10, 20), dpi=90)
        plot_connectome(t, roi_coords, edge_threshold='85%',
                        title=group + '_' + session, output_file=output_file)

def two_sample_ttest():
    """ Plots two sample ttest
    """
    n_rois = len(dataset.rois[0])
    groups = np.unique(dataset.group)
    for i in range(len(groups) - 1):
        for j in range(i + 1, len(groups)):
            pc_group_1 = pc_all[dataset.group_indices[groups[i]], :]
            pc_group_2 = pc_all[dataset.group_indices[groups[j]], :]     
            print groups[i], groups[j]
            tv, pv = ttest_ind(pc_group_1, pc_group_2)
            pv = -np.log10(pv)
    
            ind_threshold = np.where(pv < 1.3)
            pv[ind_threshold] = 0
            
            p = np.zeros((n_rois, n_rois))
            p[ind] = pv
    
            #plot connectivity matrix
            plt.figure(figsize=(8, 8))
            plt.imshow(p, interpolation='nearest', cmap=cm.hot)
            plt.colorbar()
            plt.title(groups[i] + ' / ' + groups[j] + '_' + session, fontsize=20)
            plt.xticks(range(len(roi_names)), roi_names,
                       rotation='vertical', fontsize=16)
            plt.yticks(range(len(roi_names)), roi_names, fontsize=16)
            output_file = os.path.join(set_figure_base_dir(), 'connectivity',
                                       'ttest2_connectivity_' + session + '_' + \
                                       groups[i] + ' _ ' + groups[j])
            plt.savefig(output_file)

            # plot connectome
            output_file = os.path.join(set_figure_base_dir(), 'connectivity',
                                       'ttest2_connectome_' + session + '_' + \
                                       groups[i] + ' _ ' + groups[j])

            p = (p + p.T) / 2
            plt.figure(figsize=(10, 20), dpi=90)
            plot_connectome(p, roi_coords, edge_threshold='90%',
                            title=groups[i] + ' / ' + groups[j] + '_' + session,
                            output_file=output_file)

def permuted_two_sample_ttest():
    """ plots permutation test
    """
    groups = np.unique(dataset.group)
    n_rois = len(dataset.rois[0])
    for i in range(len(groups) - 1):
        for j in range(i + 1, len(groups)):
            pc_group_1 = pc_all[dataset.group_indices[groups[i]], :]
            pc_group_2 = pc_all[dataset.group_indices[groups[j]], :]
            
            pc_groups = np.concatenate((pc_group_1, pc_group_2))
            y = np.array(pc_group_1.shape[0]*[1] + pc_group_2.shape[0]*[0])
            y = np.array([y, 1-y]).T
    
            pvals, tvals, h0 = permuted_ols(pc_groups, y,
                                            two_sided_test=True,
                                            n_perm=10000, random_state=42, n_jobs=3)
    
            pvals = pvals[:, 0]
            ind_threshold = np.where(pvals < 1.3)
            pvals[ind_threshold] = 0
            
            p = np.zeros((n_rois, n_rois))
            p[ind] = pvals
            plt.figure(figsize=(8, 8))
            plt.imshow(p, interpolation='nearest', cmap=cm.hot)
            plt.colorbar()
            plt.title(groups[i] + ' / ' + groups[j] + '_' + session, fontsize=20)
            plt.xticks(range(len(roi_names)), roi_names,
                       rotation='vertical', fontsize=16)
            plt.yticks(range(len(roi_names)), roi_names, fontsize=16)
            output_file = os.path.join(set_figure_base_dir(), 'connectivity',
                                       'ttest2_perm_connectivity_' + session + '_' + \
                                       groups[i] + ' _ ' + groups[j])
            plt.savefig(output_file)
            # plot connectome
            output_file = os.path.join(set_figure_base_dir(), 'connectivity',
                                       'ttest2_perm_connectome_' + session + '_' + \
                                       groups[i] + ' _ ' + groups[j])

            # make a symmetric matrix
            p = (p + p.T) / 2
            plt.figure(figsize=(10, 20), dpi=90)
            plot_connectome(p, roi_coords, edge_threshold='90%',
                            title=groups[i] + ' / ' + groups[j] + '_' + session,
                            output_file=output_file)


##############################################################################
# Load data and extract only
dataset = load_dynacomp()
roi_names, roi_coords = mean_coords(dataset)

# Lower diagonal
ind = np.tril_indices(len(dataset.rois[0].keys()), k=-1)
# Roi names
roi_names = sorted(dataset.rois[0].keys())

session = 'func1'
connectivity = 'pc'

# Load correlations
pc_all = load_connectivity(dataset, session, connectivity)

# z-fisher 
#pc_all = .5 * np.log((1 + pc_all)/(1 - pc_all))

one_sample_ttest()
two_sample_ttest()
permuted_two_sample_ttest()