# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 10:11:24 2015

@author: mr243268
"""

import os, time
import numpy as np
from loader import load_dynacomp, dict_to_list
from nilearn.datasets import fetch_msdl_atlas
from nilearn.input_data import NiftiMapsMasker

dataset = load_dynacomp()
atlas = fetch_msdl_atlas()

masker = NiftiMapsMasker(maps_img=atlas['maps'],
                         mask_img=dataset.mask,
                         low_pass=None,
                         high_pass=None,
                         t_r=1.05,
                         detrend=True,
                         standardize=False,
                         resampling_target='data',
                         memory_level=0,
                         verbose=5)

for i in range(len(dataset.subjects)):
    tic = time.clock()
    output_path, _ = os.path.split(dataset.func1[i])
    if not os.path.isfile(os.path.join(output_path, 'func1_msdl_no_filter.npy')):        
        print i, dataset.subjects[i]
        # add mask, filtering and detrending
        output_path, _ = os.path.split(dataset.func1[i])
        x = masker.fit_transform(dataset.func1[i])
        np.save(os.path.join(output_path, 'func1_msdl_no_filter') , x)
        x = masker.fit_transform(dataset.func2[i])
        np.save(os.path.join(output_path, 'func2_msdl_no_filter') , x)
    toc = time.clock()
    print 'time: ', toc - tic
