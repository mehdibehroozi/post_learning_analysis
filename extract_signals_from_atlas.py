# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:11:24 2015

@author: mr243268
"""

import os, time
import numpy as np
from loader import load_dynacomp
from nilearn.datasets import fetch_msdl_atlas
from nilearn.input_data import NiftiMapsMasker

dataset = load_dynacomp(preprocessing_folder='pipeline_2',
                        prefix='resampled_wr')
atlas = fetch_msdl_atlas()

# add mask, smoothing, filtering and detrending
masker = NiftiMapsMasker(maps_img=atlas['maps'],
                         mask_img=dataset.mask,
                         low_pass=.1,
                         high_pass=.01,
                         t_r=1.05,
                         smoothing_fwhm=6.,
                         detrend=True,
                         standardize=False,
                         resampling_target='data',
                         memory_level=0,
                         verbose=5)

for i in range(len(dataset.subjects)):
    tic = time.clock()
    output_path, _ = os.path.split(dataset.func1[i])
    if not os.path.isfile(os.path.join(output_path, 'func1_msdl_filter.npy')):
        print i, dataset.subjects[i]
        output_path, _ = os.path.split(dataset.func1[i])
        x = masker.fit_transform(dataset.func1[i])
        np.save(os.path.join(output_path, 'func1_msdl_filter') , x)
        x = masker.fit_transform(dataset.func2[i])
        np.save(os.path.join(output_path, 'func2_msdl_filter') , x)
    toc = time.clock()
    print 'time: ', toc - tic
