# -*- coding: utf-8 -*-
"""
Extract from each specific ROI

Created on Fri Mar 27 16:39:01 2015

@author: mehdi.rahim@cea.fr
"""
import os, time
import numpy as np
import nibabel as nib
from loader import load_dynacomp, dict_to_list
from nilearn.input_data import NiftiMapsMasker

dataset = load_dynacomp()

for i in range(len(dataset.subjects)):
    tic = time.clock()
    output_path, _ = os.path.split(dataset.func1[i])
#    if os.path.isfile(os.path.join(output_path, 'func1_rois_no_filter.npy')):        
    print i, dataset.subjects[i]
    maps_img = dict_to_list(dataset.rois[i])
    # add mask, filtering and detrending
    masker = NiftiMapsMasker(maps_img=maps_img, mask_img=dataset.mask,
                             low_pass=None,
                             high_pass=None,
                             t_r=1.05,
                             detrend=True,
                             standardize=False,
                             resampling_target='data',
                             memory_level=0,
                             verbose=5)
    output_path, _ = os.path.split(dataset.func1[i])
    x = masker.fit_transform(dataset.func1[i])
    np.save(os.path.join(output_path, 'func1_rois_no_filter') , x)
#    if os.path.isfile(os.path.join(output_path, 'func2_rois_no_filter.npy')):
    print i, dataset.subjects[i]
    maps_img = dict_to_list(dataset.rois[i])
    # add mask, filtering and detrending
    masker = NiftiMapsMasker(maps_img=maps_img, mask_img=dataset.mask,
                             low_pass=None,
                             high_pass=None,
                             t_r=1.05,
                             detrend=True,
                             standardize=False,
                             resampling_target='data',
                             memory_level=0,
                             verbose=5)

    x = masker.fit_transform(dataset.func2[i])
    np.save(os.path.join(output_path, 'func2_rois_no_filter') , x)
    toc = time.clock()
    print toc - tic
