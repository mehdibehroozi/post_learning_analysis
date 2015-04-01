# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:22:56 2015

@author: mr243268
"""
import os
import numpy as np
import nibabel as nib
from loader import load_dynacomp

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


for subject_id in dataset.subjects:
    compute_rois_coords(subject_id)