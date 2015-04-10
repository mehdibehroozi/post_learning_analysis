# -*- coding: utf-8 -*-
"""
Extract from each specific ROI

Created on Fri Mar 27 16:39:01 2015

@author: mehdi.rahim@cea.fr
"""
import os, time
import numpy as np
from loader import load_dynacomp, dict_to_list
from nilearn.input_data import NiftiMapsMasker


dataset = load_dynacomp(preprocessing_folder='pipeline_2',
                        prefix='resampled_wr')

# func1, func2
for idx, func in enumerate([dataset.func1, dataset.func2]):
    # all the subjects
    for i in range(len(dataset.subjects)):
        tic = time.clock()
        output_path, _ = os.path.split(func[i])
        print dataset.subjects[i]
        maps_img = dict_to_list(dataset.rois[i])
        # add mask, smoothing, filtering and detrending
        masker = NiftiMapsMasker(maps_img=maps_img, mask_img=dataset.mask,
                                 low_pass=.1,
                                 high_pass=.01,
                                 smoothing_fwhm=6.,
                                 t_r=1.05,
                                 detrend=True,
                                 standardize=False,
                                 resampling_target='data',
                                 memory_level=0,
                                 verbose=5)
        output_path, _ = os.path.split(func[i])
        # extract the signal to x
        x = masker.fit_transform(func[i])
        np.save(os.path.join(output_path,
                             'func' + str(idx+1) + '_rois_filter'), x)
        toc = time.clock()
        print toc - tic
