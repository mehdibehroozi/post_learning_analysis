# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:30:42 2015

@author: mr243268
"""
import os
import numpy as np
from loader import load_dynacomp, load_dynacomp_roi_timeseries, \
                   load_roi_names_and_coords, set_figure_base_dir, set_data_base_dir
import matplotlib.pyplot as plt
from matplotlib import cm
from nilearn.plotting import plot_connectome
from sklearn import covariance
from nilearn.group_sparse_covariance import GroupSparseCovarianceCV



def plot_connectivity_matrix(subject_id, group, pc, roi_names,
                             suffix, session='func1', save=True):
    """Plots connectivity matrix of pc
    """
    title = '-'.join([suffix, group, subject_id, session])
    # plot matrix
    output_file = os.path.join(set_figure_base_dir('connectivity'),
                               '_'.join([suffix, 'matrix', group,
                                         session, subject_id]))
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
    print session
    title = '-'.join([suffix, group, subject_id, session])
    output_file = os.path.join(set_figure_base_dir('connectivity'),
                               '_'.join([suffix, 'connectome', group,
                                         session, subject_id]))
    plt.figure(figsize=(10, 20), dpi=90)
    if save:    
        plot_connectome(pc, roi_coords, edge_threshold='85%', title=title,
                        output_file=output_file)
    else:
        plot_connectome(pc, roi_coords, edge_threshold='85%', title=title)
    
def compute_pearson_connectivity(subject_id, group, session='func1',
                                 preprocessing_folder='pipeline_1',
                                 plot=True, save=True, save_file=True):
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
        print session
        plot_connectivity_matrix(subject_id, group, pc,
                                 roi_names, 'pc',
                                 session, save)
        plot_connectivity_glassbrain(subject_id, group, pc,
                                     roi_coords, 'pc', session, save)
    if save_file:
        CONN_DIR = set_data_base_dir('Dynacomp/connectivity')
        if not os.path.isdir(os.path.join(CONN_DIR, subject_id)):
            os.mkdir(os.path.join(CONN_DIR, subject_id))
        output_file = os.path.join(CONN_DIR, subject_id, 'pc_' + session)
        np.savez(output_file, correlation=pc,
                 roi_names=roi_names, roi_coords=roi_coords)

    return pc, roi_names, roi_coords


def compute_graph_lasso_covariance(subject_id, group, session='func1',
                                   preprocessing_folder='pipeline_1',
                                   plot=True, save=True, save_file=True):
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
    if plot:
        plot_connectivity_matrix(subject_id, group, gl.covariance_,
                                 roi_names, 'gl_covariance', session, save)
        plot_connectivity_matrix(subject_id, group, gl.precision_,
                                 roi_names, 'gl_precision', session, save)
        sparsity = (gl.precision_ == 0)
        plot_connectivity_matrix(subject_id, group, sparsity,
                                 roi_names, 'gl_sparsity', session, save)

        plot_connectivity_glassbrain(subject_id, group, gl.covariance_,
                                     roi_coords, 'gl_covariance', session, save)

    if save_file:
        CONN_DIR = set_data_base_dir('Dynacomp/connectivity')
        sparsity = (gl.precision_ == 0)
        if not os.path.isdir(os.path.join(CONN_DIR, subject_id)):
            os.mkdir(os.path.join(CONN_DIR, subject_id))
        output_file = os.path.join(CONN_DIR, subject_id, 'gl_' + session)
        np.savez(output_file, covariance=gl.covariance_,
                 precision=gl.precision_, sparsity=sparsity,
                 roi_names=roi_names, roi_coords=roi_coords)
    return gl, roi_names, roi_coords


def compute_group_sparse_covariance(dataset, session='func1',
                                    preprocessing_folder='pipeline_1',
                                    plot=True, save=True, save_file=True):
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
            roi_names,\
            roi_coords = load_roi_names_and_coords(dataset.subjects[i])

            plot_connectivity_matrix(dataset.subjects[i], dataset.group[i],
                                     gsc.covariances_[..., i],
                                     roi_names, 'gsc_covariance', session, save)

            plot_connectivity_matrix(dataset.subjects[i], dataset.group[i],
                                     gsc.precisions_[..., i],
                                     roi_names, 'gsc_precision', session, save)

            sparsity = (gsc.precisions_[..., i] == 0)
            plot_connectivity_matrix(dataset.subjects[i], dataset.group[i],
                                     sparsity,
                                     roi_names, 'gsc_sparsity', session, save)

            plot_connectivity_glassbrain(dataset.subjects[i], dataset.group[i],
                                         gsc.covariances_[..., i],
                                         roi_coords, 'gsc_covariance', session, save)

    for i in range(len(dataset.subjects)):
        # load rois
        roi_names,\
        roi_coords = load_roi_names_and_coords(dataset.subjects[i])
        sparsity = (gsc.precisions_[..., i] == 0)            
        CONN_DIR = set_data_base_dir('Dynacomp/connectivity')
        subject_id = dataset.subjects[i]
        if not os.path.isdir(os.path.join(CONN_DIR, subject_id)):
            os.mkdir(os.path.join(CONN_DIR, subject_id))
        output_file = os.path.join(CONN_DIR, subject_id, 'gsc_' + session)
        np.savez(output_file, covariance=gsc.covariances_[..., i],
                 precision=gsc.precisions_[..., i], sparsity=sparsity,
                 roi_names=roi_names, roi_coords=roi_coords)

    return gsc, roi_names, roi_coords

##############################################################################
dataset = load_dynacomp()

for session_i in ['func1', 'func2']:
    for i in range(len(dataset.subjects)):
        print dataset.subjects[i], session_i
#        compute_pearson_connectivity(dataset.subjects[i], dataset.group[i], plot=False,
#                                     save=False, session=session_i, save_file=False)
        compute_graph_lasso_covariance(dataset.subjects[i], dataset.group[i], plot=True,
                                       save=False, session=session_i, save_file=False)
        break
#    compute_group_sparse_covariance(dataset, save=False, plot=False,
#                                    session=session_i, save_file=False)