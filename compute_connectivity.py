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
from nilearn.group_sparse_covariance import GroupSparseCovarianceCV

def plot_connectivity_matrix(subject_id, group, pc, roi_names,
                             suffix, session='func1', save=True):
    """Plots connectivity matrix of pc
    """
    title = subject_id + ' - ' + group
    # plot matrix
    output_file = os.path.join(set_figure_base_dir('connectivity'),
                               '_'.join([suffix, 'matrix', group,
                                         str(session), subject_id]))
    plt.figure(figsize=(8, 8))
    plt.imshow(pc, cmap=cm.bwr, interpolation='nearest',
               vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(roi_names)), roi_names,
               rotation='vertical', fontsize=16)
    plt.yticks(range(len(roi_names)), roi_names, fontsize=16)
    plt.title(title, fontsize=20)
    plt.tight_layout()
    if save:
        plt.savefig(output_file)


def plot_connectivity_glassbrain(subject_id, group, pc, roi_coords,
                                 suffix, session='func1', save=True):
    """Plots connectome of pc
    """
    title = subject_id + ' - ' + group
    output_file = os.path.join(set_figure_base_dir('connectivity'),
                               '_'.join([suffix, 'connectome', group,
                                         str(session), subject_id]))
    plt.figure(figsize=(10, 20), dpi=90)
    if save:    
        plot_connectome(pc, roi_coords, edge_threshold='85%', title=title,
                        output_file=output_file)
    else:
        plot_connectome(pc, roi_coords, edge_threshold='85%', title=title)
    
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
        plot_connectivity_matrix(subject_id, group, pc,
                                 roi_names, 'pearson_corr', save)
        plot_connectivity_glassbrain(subject_id, group, pc,
                                     roi_coords, 'pearson_corr', save)

    return pc, roi_names, roi_coords


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
        plot_connectivity_matrix(subject_id, group, gl.covariance_,
                                 roi_names, 'gl_covariance', save)
        plot_connectivity_matrix(subject_id, group, gl.precision_,
                                 roi_names, 'gl_precision', save)
        sparsity = (gl.precision_ == 0)
        plot_connectivity_matrix(subject_id, group, sparsity,
                                 roi_names, 'gsc_sparsity', save)


        plot_connectivity_glassbrain(subject_id, group, gl.covariance_,
                                     roi_coords, 'gl_covariance', save)

    return pc, roi_names, roi_coords


def compute_group_sparse_covariance(dataset, session='func1',
                                    preprocessing_folder='pipeline_1',
                                    plot=True, save=True):
    """Returns Group sparse covariance for all subjects
    """
    ts = []
    for subject_id in dataset.subjects:
        ts.append(load_dynacomp_roi_timeseries(subject_id, session=session,\
                  preprocessing_folder=preprocessing_folder))
    gsc = GroupSparseCovarianceCV(verbose=2)
    gsc.fit(ts)

    if plot:
        for i in range(len(dataset.subjects)):
            # load rois
            roi_names, roi_coords = load_roi_names_and_coords(dataset.subjects[i])

            plot_connectivity_matrix(dataset.subjects[i], dataset.group[i],
                                     gsc.covariances_[..., i],
                                     roi_names, 'gsc_covariance', save)

            plot_connectivity_matrix(dataset.subjects[i], dataset.group[i],
                                     gsc.precisions_[..., i],
                                     roi_names, 'gsc_precision', save)

            sparsity = (gsc.precisions_[..., i] == 0)
            plot_connectivity_matrix(dataset.subjects[i], dataset.group[i],
                                     sparsity,
                                     roi_names, 'gsc_sparsity', save)

            plot_connectivity_glassbrain(dataset.subjects[i], dataset.group[i],
                                         gsc.covariances_[..., i],
                                         roi_coords, 'gsc_covariance', save)
    return gsc



##############################################################################
dataset = load_dynacomp()

for i in range(len(dataset.subjects)):
    print dataset.subjects[i]
    pc, roi_names, roi_coords = compute_pearson_connectivity(dataset.subjects[i],
                                                             dataset.group[i])    
    compute_graph_lasso_covariance(dataset.subjects[i], dataset.group[i])

compute_group_sparse_covariance(dataset)