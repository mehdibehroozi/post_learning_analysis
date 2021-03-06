# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:30:42 2015

@author: mr243268
"""
import os
import numpy as np
from loader import load_dynacomp, load_dynacomp_roi_timeseries, \
                   load_roi_names_and_coords, set_figure_base_dir,\
                   set_data_base_dir, load_dynacomp_msdl_timeseries, \
                   load_msdl_names_and_coords
import matplotlib.pyplot as plt
from matplotlib import cm
from nilearn.plotting import plot_connectome
from sklearn import covariance
from nilearn.group_sparse_covariance import GroupSparseCovarianceCV


def plot_connectivity_matrix(subject_id, group, pc, roi_names,
                             suffix, session='func1',
                             preprocessing_folder='pipeline_1',
                             save=True, msdl=False):
    """Plots connectivity matrix of pc
    """
    title = '-'.join([suffix, group, subject_id, session])
    # plot matrix
    output_folder = os.path.join(set_figure_base_dir('connectivity'), suffix)
    
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
   
    output_file = os.path.join(output_folder,
                               '_'.join([suffix, 'matrix', group,
                                         session,
                                         preprocessing_folder, subject_id]))

    if msdl:
        title += '_msdl'
        output_file += '_msdl'

    if msdl:
        plt.figure(figsize=(12, 12))
    else:
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
                                 suffix, session='func1',
                                 preprocessing_folder='pipeline_1',
                                 save=True, msdl=False):
    """Plots connectome of pc
    """
    title = '-'.join([suffix, group, subject_id, session])
    
    output_folder = os.path.join(set_figure_base_dir('connectivity'), suffix)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    output_file = os.path.join(output_folder,
                               '_'.join([suffix, 'connectome', group,
                                         session,
                                         preprocessing_folder, subject_id]))
                                         
    if msdl:
        title += '_msdl'
        output_file += '_msdl'
                                         
    plt.figure(figsize=(10, 20), dpi=90)
    if save:    
        plot_connectome(pc, roi_coords, edge_threshold='90%', title=title,
                        output_file=output_file)
    else:
        plot_connectome(pc, roi_coords, edge_threshold='90%', title=title)

def compute_pearson_connectivity(subject_id, group, session='func1',
                                 preprocessing_folder='pipeline_1',
                                 plot=True, save=True, save_file=True,
                                 msdl=False):
    """Returns Pearson correlation coefficient for a subject_id
    """    
    # load timeseries
    if msdl:
        ts = load_dynacomp_msdl_timeseries(subject_id, session=session,
                                          preprocessing_folder=preprocessing_folder)
        roi_names, roi_coords = load_msdl_names_and_coords()
    else:
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
                                 session,
                                 preprocessing_folder, save, msdl)
        plot_connectivity_glassbrain(subject_id, group, pc,
                                     roi_coords, 'pc', session,
                                     preprocessing_folder, save, msdl)
    if save_file:
        CONN_DIR = set_data_base_dir('Dynacomp/connectivity')
        if not os.path.isdir(os.path.join(CONN_DIR, subject_id)):
            os.mkdir(os.path.join(CONN_DIR, subject_id))
        output_file = os.path.join(CONN_DIR, subject_id,
                                   'pc_' + session + '_' + preprocessing_folder)
        if msdl:
            output_file += '_msdl'
        np.savez(output_file, correlation=pc,
                 roi_names=roi_names, roi_coords=roi_coords)
    return pc, roi_names, roi_coords


def compute_graph_lasso_covariance(subject_id, group, session='func1',
                                   preprocessing_folder='pipeline_1',
                                   plot=True, save=True, save_file=True,
                                   msdl=False):
    """Returns graph lasso covariance for a subject_id
    """
    # load timeseries
    if msdl:
        ts = load_dynacomp_msdl_timeseries(subject_id, session=session,
                                          preprocessing_folder=preprocessing_folder)
        roi_names, roi_coords = load_msdl_names_and_coords()
    else:
        ts = load_dynacomp_roi_timeseries(subject_id, session=session,
                                          preprocessing_folder=preprocessing_folder)
        # load rois
        roi_names, roi_coords = load_roi_names_and_coords(subject_id)

    # compute covariance
    gl = covariance.GraphLassoCV(verbose=2)
    gl.fit(ts)
    if plot:
        plot_connectivity_matrix(subject_id, group, gl.covariance_,
                                 roi_names, 'gl_covariance', session,
                                 preprocessing_folder, save, msdl)
        plot_connectivity_matrix(subject_id, group, gl.precision_,
                                 roi_names, 'gl_precision', session,
                                 preprocessing_folder, save, msdl)
        sparsity = (gl.precision_ == 0)
        plot_connectivity_matrix(subject_id, group, sparsity,
                                 roi_names, 'gl_sparsity', session,
                                 preprocessing_folder, save, msdl)

        plot_connectivity_glassbrain(subject_id, group, gl.covariance_,
                                     roi_coords, 'gl_covariance', session,
                                     preprocessing_folder, save, msdl)

    if save_file:
        CONN_DIR = set_data_base_dir('Dynacomp/connectivity')
        sparsity = (gl.precision_ == 0)
        if not os.path.isdir(os.path.join(CONN_DIR, subject_id)):
            os.mkdir(os.path.join(CONN_DIR, subject_id))
        output_file = os.path.join(CONN_DIR, subject_id,
                                   'gl_' + session + '_' + preprocessing_folder)
        if msdl:
            output_file += '_msdl'
        np.savez(output_file, covariance=gl.covariance_,
                 precision=gl.precision_, sparsity=sparsity,
                 roi_names=roi_names, roi_coords=roi_coords)
    return gl, roi_names, roi_coords


def compute_group_sparse_covariance(dataset, session='func1',
                                    preprocessing_folder='pipeline_1',
                                    plot=True, save=True, save_file=True,
                                    msdl=False):
    """Returns Group sparse covariance for all subjects
    """

    ts = []
    # load timeseries
    if msdl:
        for subject_id in dataset.subjects:
            ts.append(load_dynacomp_msdl_timeseries(subject_id,\
                      session=session, preprocessing_folder=preprocessing_folder))
        roi_names, roi_coords = load_msdl_names_and_coords()
    else:
        for subject_id in dataset.subjects:
            ts.append(load_dynacomp_roi_timeseries(subject_id, session=session,\
                      preprocessing_folder=preprocessing_folder))
        # load rois
        roi_names, roi_coords = load_roi_names_and_coords(subject_id)

    gsc = GroupSparseCovarianceCV(verbose=2)
    gsc.fit(ts)

    if plot:
        for i in range(len(dataset.subjects)):
            if not msdl:
                # load rois
                roi_names,\
                roi_coords = load_roi_names_and_coords(dataset.subjects[i])

            plot_connectivity_matrix(dataset.subjects[i], dataset.group[i],
                                     gsc.covariances_[..., i],
                                     roi_names, 'gsc_covariance', session,
                                     preprocessing_folder, save, msdl)

            plot_connectivity_matrix(dataset.subjects[i], dataset.group[i],
                                     gsc.precisions_[..., i],
                                     roi_names, 'gsc_precision', session,
                                     preprocessing_folder, save, msdl)

            sparsity = (gsc.precisions_[..., i] == 0)
            plot_connectivity_matrix(dataset.subjects[i], dataset.group[i],
                                     sparsity,
                                     roi_names, 'gsc_sparsity', session,
                                     preprocessing_folder, save, msdl)

            plot_connectivity_glassbrain(dataset.subjects[i], dataset.group[i],
                                         gsc.covariances_[..., i],
                                         roi_coords, 'gsc_covariance', session,
                                         preprocessing_folder, save, msdl)

    for i in range(len(dataset.subjects)):
        if not msdl:
            # load rois
            roi_names,\
            roi_coords = load_roi_names_and_coords(dataset.subjects[i])
        sparsity = (gsc.precisions_[..., i] == 0)            
        CONN_DIR = set_data_base_dir('Dynacomp/connectivity')
        subject_id = dataset.subjects[i]
        if not os.path.isdir(os.path.join(CONN_DIR, subject_id)):
            os.mkdir(os.path.join(CONN_DIR, subject_id))
        output_file = os.path.join(CONN_DIR, subject_id,
                                   'gsc_' + session + '_' + preprocessing_folder)
        if msdl:
            output_file += '_msdl'
        np.savez(output_file, covariance=gsc.covariances_[..., i],
                 precision=gsc.precisions_[..., i], sparsity=sparsity,
                 roi_names=roi_names, roi_coords=roi_coords)

    return gsc, roi_names, roi_coords

##############################################################################

preprocessing_folder = 'pipeline_1'
prefix = 'swr'
#preprocessing_folder = 'pipeline_2'
#prefix = 'resampled_wr'
msdl = False

dataset = load_dynacomp(preprocessing_folder=preprocessing_folder,
                        prefix=prefix)

for session_i in ['func1', 'func2']:
    for i in range(len(dataset.subjects)):
        print dataset.subjects[i], session_i
        compute_pearson_connectivity(dataset.subjects[i],
                                     dataset.group[i],
                                     session=session_i,
                                     preprocessing_folder=preprocessing_folder,
                                     plot=True,
                                     save=True,
                                     save_file=True,
                                     msdl=msdl)


        compute_graph_lasso_covariance(dataset.subjects[i],
                                       dataset.group[i],
                                       session=session_i,
                                       preprocessing_folder=preprocessing_folder,
                                       plot=True,
                                       save=True,
                                       save_file=True,
                                       msdl=msdl)

    compute_group_sparse_covariance(dataset,
                                    session=session_i,
                                    preprocessing_folder=preprocessing_folder,
                                    plot=True,
                                    save=True,
                                    save_file=True,
                                    msdl=msdl)