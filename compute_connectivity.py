# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:30:42 2015

@author: mr243268
"""
import os
import numpy as np
from loader import load_dynacomp, load_dynacomp_roi_timeseries, \
                   load_roi_names_and_coords, set_figure_base_dir
import matplotlib.pyplot as plt
from matplotlib import cm
from nilearn.plotting import plot_connectome
from sklearn import covariance

def plot_connectivity_matrix(subject_id, group, pc, roi_names, suffix):
        title = subject_id + ' - ' + group
        # plot matrix
        output_file = os.path.join(set_figure_base_dir('connectivity'),
                                   '_'.join([suffix, 'matrix', group, '1', subject_id]))
        plt.figure(figsize=(8, 8))
        plt.imshow(pc, cmap=cm.bwr, interpolation='nearest',
                   vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(roi_names)), roi_names,
                   rotation='vertical', fontsize=16)
        plt.yticks(range(len(roi_names)), roi_names, fontsize=16)
        plt.title(title, fontsize=20)
        plt.tight_layout()
        plt.savefig(output_file)


def plot_connectivity_glassbrain(subject_id, group, pc, roi_coords, suffix):
        title = subject_id + ' - ' + group
        output_file = os.path.join(set_figure_base_dir('connectivity'),
                                   '_'.join([suffix, 'connectome', group, '1', subject_id]))
        plt.figure(figsize=(10, 20), dpi=90)
        plot_connectome(pc, roi_coords, edge_threshold='85%', title=title, output_file=output_file)


def compute_graph_lasso_covariance(subject_id, group, session='func1',
                                   preprocessing_folder='pipeline_1',
                                   plot=True, save=True):
    """Returns graph lasso covariance for a subject_id
    """
    # load timeseries
    ts = load_dynacomp_roi_timeseries(subject_id, session=session,
                                      preprocessing_folder=preprocessing_folder)
    # load rois
    roi_names, roi_coords = load_roi_names_and_coords(subject_id)

    # compute covariance
    gl = covariance.GraphLassoCV(verbose=2)
    gl.fit(ts)
    pc = gl.covariance_
    if plot:
        plot_connectivity_matrix(subject_id, group, pc, roi_names, 'gl_covariance')
        plot_connectivity_glassbrain(subject_id, group, pc, roi_coords, 'gl_covariance')

    return pc, roi_names, roi_coords

    
def compute_pearson_connectivity(subject_id, group, session='func1',
                                 preprocessing_folder='pipeline_1',
                                 plot=True, save=True):
    """Returns Pearson correlation coefficient for a subject_id
    """
    
    # load timeseries
    ts = load_dynacomp_roi_timeseries(subject_id, session=session,
                                      preprocessing_folder=preprocessing_folder)
    # load rois
    roi_names, roi_coords = load_roi_names_and_coords(subject_id)

    # pearson correlation    
    pc = np.corrcoef(ts.T)

    if plot:
        plot_connectivity_matrix(subject_id, group, pc, roi_names, 'pearson_corr')
        plot_connectivity_glassbrain(subject_id, group, pc, roi_coords, 'pearson_corr')

    return pc, roi_names, roi_coords



##############################################################################
dataset = load_dynacomp()

for i in range(len(dataset.subjects)):
    print dataset.subjects[i]
    pc, roi_names, roi_coords = compute_pearson_connectivity(dataset.subjects[i],
                                                             dataset.group[i])    
    compute_graph_lasso_covariance(dataset.subjects[i], dataset.group[i])
