# -*- coding: utf-8 -*-
"""
Extract from each specific ROI

Created on Fri Mar 27 16:39:01 2015

@author: mehdi.rahim@cea.fr
"""

from loader import load_dynacomp, list_of_dicts_to_key_list, dict_to_list
from nilearn.input_data import NiftiMapsMasker, NiftiMasker


dataset = load_dynacomp()

