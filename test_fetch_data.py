# -*- coding: utf-8 -*-
"""
Script to test functions

Created on Thu Mar 26 15:02:11 2015

@author: mehdi.rahim@cea.fr
"""

from loader import load_dynacomp, list_of_dicts_to_key_list, dict_to_list
from nilearn.input_data import NiftiMapsMasker
import time

# Load Dynacomp dataset
dataset = load_dynacomp()

# Dataset keys
print 'keys\n', dataset.keys()

# Dataset functional 1
print 'func1\n', dataset.func1

# Dataset behaviordata : prePerf
print 'prePerf\n', list_of_dicts_to_key_list(dataset.behavior, 'prePerf')

# Generate seed-masker for subject 0
maps_img = dict_to_list(dataset.rois[1])

tic = time.clock()
masker = NiftiMapsMasker(maps_img, verbose=5)
x = masker.fit_transform(dataset.func1[1])
toc = time.clock()
print toc - tic