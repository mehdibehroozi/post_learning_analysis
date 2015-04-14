# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 21:27:50 2015

@author: mr243268
"""

import os
import numpy as np
from loader import load_dynacomp, load_msdl_names_and_coords,\
                   load_dynacomp_fc, load_roi_names_and_coords,\
                   set_figure_base_dir, load_dynacomp_roi_timeseries,\
                   list_of_dicts_to_key_list
import matplotlib.pyplot as plt

preprocessing_folder='pipeline_1'
prefix='swr'

dataset = load_dynacomp(preprocessing_folder, prefix)

# Behavior data
behav_data = dataset.behavior
# Add deltas
for i in range(len(behav_data)):
    for key in ['Thresh', 'RT', 'HIT_RT', 'Perf', 'Conf_mean']:
        behav_data[i]['delta' + key] = - behav_data[i]['post' + key] + \
                                       behav_data[i]['pre' + key]


subject_id = dataset.subjects[0]
all_pc = []
avn = []
v = []
av = []
gr_ind = []
for i, subject_id in enumerate(dataset.subjects):
    ts = load_dynacomp_roi_timeseries(subject_id, session='func1',
                                      preprocessing_folder=preprocessing_folder)                   
    # ROIs : 19, 21
    pc = np.corrcoef(ts[:,19], ts[:,21])[0,1]
    if dataset.group[i] == 'v':
        v.append(pc)
        gr_ind.append(i)
    if dataset.group[i] == 'avn':
        avn.append(pc)
        gr_ind.append(i)
    if dataset.group[i] == 'av':
        av.append(pc)
    all_pc.append(pc)

plt.figure()
plt.boxplot([avn, v, av])
plt.xticks(range(1, 4), ['avn', 'v', 'av'], fontsize=16)
plt.grid(axis='y')
plt.title('pSTS_rh-precuneus_rh Connectivity', fontsize=16)
plt.savefig('psts_precuneus.png')


#gr_ind = np.array(gr_ind)
#all_pc = np.array(all_pc)
#
#import seaborn as sns
#for key in behav_data[0].keys():
#    b = list_of_dicts_to_key_list(behav_data, key)
#    b = np.array(b)
#    sns.jointplot(all_pc[gr_ind], b[gr_ind])
#    sns.axlabel('pc', key, fontsize=16)
