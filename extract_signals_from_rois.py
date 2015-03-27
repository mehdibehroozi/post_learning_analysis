# -*- coding: utf-8 -*-
"""
Extract from each specific ROI

Created on Fri Mar 27 16:39:01 2015

@author: mehdi.rahim@cea.fr
"""
import os
import numpy as np
import nibabel as nib
from loader import load_dynacomp, list_of_dicts_to_key_list, dict_to_list
from nilearn.input_data import NiftiMapsMasker, NiftiMasker
from nilearn.masking import apply_mask

dataset = load_dynacomp()

def compute_rois_coords(subject_id):
    # for each subject
    for rois in dataset.rois:
        
        roi_center = {}
        # for each roi
        for key in rois.keys():
            affine = nib.load(rois[key]).get_affine()
            data = nib.load(rois[key]).get_data()
            centroid = np.mean(np.where(data==1),axis=1)
            centroid = np.append(centroid, 1)
            centroid_mni = np.dot(affine, centroid)[:-1]
            roi_center[key] = centroid_mni
        filepath, _ = os.path.split(rois[key])
        print filepath
        np.save(os.path.join(filepath, 'rois_coords.npy') , roi_center)