# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:12:05 2015

@author: mehdi.rahim@cea.fr
"""

import numpy as np
from loader import load_dynacomp, list_of_dicts_to_key_list
import matplotlib.pyplot as plt  
                   
# Load dataset                   
dataset = load_dynacomp()
groups = ['av', 'v', 'avn']
# Behavior data
behav_data = dataset.behavior
# Add deltas
for i in range(len(behav_data)):
    for key in ['Thresh', 'RT', 'HIT_RT', 'Perf', 'Conf_mean']:
        behav_data[i]['delta' + key] = behav_data[i]['post' + key] - \
                                       behav_data[i]['pre' + key]

# for each behav score
for key in behav_data[0].keys():
    scores = []
    # for each group
    for group in groups:
        bd = np.array(behav_data)
        scores.append(list_of_dicts_to_key_list(\
                      bd[dataset.group_indices[group]], key))
    plt.figure()
    plt.boxplot(scores)
    plt.xticks(range(1,4), groups, fontsize=16)
    plt.title(key, fontsize=16)
    plt.grid(axis='y')
