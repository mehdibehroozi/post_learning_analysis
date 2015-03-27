# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 17:01:31 2015

@author: mehdi.rahim@cea.fr
"""


from loader import load_dynacomp
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from nilearn.plotting import plot_roi

CACHE_DIR = '/disk4t/mehdi/data/tmp'
dataset = load_dynacomp()


masker = MultiNiftiMasker(mask_strategy='epi', memory=CACHE_DIR,
                          memory_level=2, n_jobs=10, verbose=5)

#masker = NiftiMasker(mask_strategy='epi', memory=CACHE_DIR,
#                     memory_level=5, verbose=5)
           
imgs = dataset.func1 + dataset.func2
masker.fit(imgs)

plot_roi(masker.mask_img_)