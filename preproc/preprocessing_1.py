# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 13:48:51 2015

@author: mehdi.rahim@cea.fr
"""


from pypreprocess.nipype_preproc_spm_utils import do_subjects_preproc

jobfile = 'pipeline_1.ini'

# preprocess the data
results = do_subjects_preproc(jobfile)