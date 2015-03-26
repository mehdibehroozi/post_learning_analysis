# -*- coding: utf-8 -*-
"""

Data loading functions of Dynacomp dataset,
and some utils.

Created on Thu Mar 26 14:09:41 2015

@author: mehdi.rahim@cea.fr
"""
import os, glob
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.datasets.base import Bunch

def set_base_dir():
    """ base_dir
    """
    base_dir = ''
    with open(os.path.join(os.path.dirname(__file__), 'paths.pref'),
              'rU') as f:
        paths = [x.strip() for x in f.read().split('\n')]
        for path in paths:
            if os.path.isdir(path):
                base_dir = path
                break
    if base_dir == '':
        raise OSError('Data not found !')
    return base_dir

def set_data_base_dir(folder):
    """ base_dir + folder
    """
    return os.path.join(set_base_dir(), folder)

def set_group_indices(group):
    """Returns indices for each clinical group
    """
    group = np.array(group)
    idx = {}
    for g in ['av', 'v', 'avn']:
        idx[g] = np.where(group == g)[0]
    return idx

def load_dynacomp_rois():
    """ Returns paths of Dynacomp ROIs
    """
    ROI_DIR = set_data_base_dir('Dynacomp/rois')
    
    subject_paths = sorted(glob.glob(os.path.join(ROI_DIR, '[A-Z][A-Z]*')))
    subject_rois = []
    for f in subject_paths:
        # subject id
        _, subject_id = os.path.split(f)
        roi_files = sorted(glob.glob(os.path.join(f, '*.nii')))
        rois_dict = {}
        for r in roi_files:
            _, roi_name = os.path.split(r)
            roi_name, _ = os.path.splitext(roi_name)
            rois_dict[roi_name] = r
        subject_rois.append(rois_dict)
    return subject_rois

def load_dynacomp(preprocessing_folder='pipeline_1', prefix='swr'):
    """ Returns paths of Dynacomp preprocessed resting-state fMRI
    """
    BASE_DIR = set_data_base_dir('Dynacomp')
    SUBJ_DIR = os.path.join(BASE_DIR, 'preprocessed', preprocessing_folder)
    subject_paths = sorted(glob.glob(os.path.join(SUBJ_DIR, '[A-Z][A-Z]*')))
    description = pd.read_csv(os.path.join(BASE_DIR, 'subject_infos.csv'))
    session1_files = []
    session2_files = []
    anat_files = []
    group = []
    subjects = []
    for f in subject_paths:
        # subject id
        _, subject_id = os.path.split(f)
        # set prefix
        # functional data
        session1_files.append(glob.glob(os.path.join(f, 'fMRI', 'acquisition1',
                                                     prefix + 'rest1*.nii'))[0])
        session1_files.append(glob.glob(os.path.join(f, 'fMRI', 'acquisition1',
                                                     prefix + 'rest2*.nii'))[0])
        # anatomical data
        anat_files.append(glob.glob(os.path.join(f, 't1mri', 'acquisition1',
                                                 'wanat*.nii'))[0])
        # subject group
        gr = description[description.NIP == subject_id].GROUP.values
        if len(gr) > 0:
            group.append(gr[0])
        # subject id
        subjects.append(subject_id)
    
    indices = set_group_indices(group)
    rois = load_dynacomp_rois()
    return Bunch(func1=session1_files,
                 func2=session2_files,
                 anat=anat_files, group_indices=indices, rois=rois,
                 group=group, subjects=subjects)
 
def array_to_niis(data, mask):
    """ Converts masked nii 4D array to 4D niimg
    """
    mask_img = nib.load(mask)
    data_ = np.zeros(data.shape[:1] + mask_img.shape)
    data_[:, mask_img.get_data().astype(np.bool)] = data
    data_ = np.transpose(data_, axes=(1, 2, 3, 0))
    return nib.Nifti1Image(data_, mask_img.get_affine())

def array_to_nii(data, mask):
    """ Converts masked nii 3D array to 3D niimg
    """
    mask_img = nib.load(mask)
    data_ = np.zeros(mask_img.shape)
    data_[mask_img.get_data().astype(np.bool)] = data
    return nib.Nifti1Image(data_, mask_img.get_affine())