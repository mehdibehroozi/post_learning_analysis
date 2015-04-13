# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 17:29:17 2015

@author: mehdi.rahim@cea.fr
"""

from loader import load_dynacomp, load_dynacomp_msdl_timeseries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

session = 'func1'

dataset = load_dynacomp(preprocessing_folder='pipeline_2', prefix='resampled_wr')

for subject_id in dataset.subjects:
    msdl1 = load_dynacomp_msdl_timeseries(subject_id=subject_id,
                                         session=session,
                                         preprocessing_folder='pipeline_1')
    msdl2 = load_dynacomp_msdl_timeseries(subject_id=subject_id,
                                         session=session,
                                         preprocessing_folder='pipeline_2')
                                         

    for i in range(msdl1.shape[1]):
        plt.figure()
        plt.plot(StandardScaler().fit_transform(msdl1[:100, i]))
        plt.plot(StandardScaler().fit_transform(msdl2[:100, i]))
    break
