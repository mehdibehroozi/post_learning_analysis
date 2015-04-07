# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:37:52 2015

@author: mr243268
"""

import os
import numpy as np
from loader import load_dynacomp, set_data_base_dir,\
                    load_msdl_names_and_coords, load_dynacomp_fc
from nilearn.plotting import plot_connectome
import seaborn as sns
import matplotlib.pyplot as plt

STAT_DIR = set_data_base_dir('Dynacomp/stat')

def read_test(metric, group, session):
    """Returns test data
    """
    filename = '_'.join(['ttest_connectome', session, group, metric, 'msdl' ])
    path = os.path.join(STAT_DIR, filename + '.npy')
    return np.load(path)


def read_test2(metric, groups, session):
    """Returns test data
    """
    filename = '_'.join(['ttest2_connectome', session, groups[0], groups[1], metric, 'msdl' ])
    path = os.path.join(STAT_DIR, filename + '.npy')
    return np.load(path)

dataset = load_dynacomp()
behav_data = dataset.behavior
behav_keys = ['Perf', 'Thresh', 'RT']


roi_names, roi_coords = load_msdl_names_and_coords()
stat_av = read_test('pc', 'av', 'avg')
stat_v = read_test('pc', 'v', 'avg')
stat2 = read_test2('pc', ['av', 'v'], 'avg')


i, j = np.unravel_index(stat2.argmax(), stat2.shape)
print 'av 1sample pval :', stat_av[i, j]
print 'v 1sample pval :', stat_v[i, j]
print roi_names[i], roi_names[j]
print i, j

m = np.eye(2)
m[1,0] = stat2[i, j]
m[0,1] = m[1,0] 
plot_connectome(m, [roi_coords[i], roi_coords[j]])


conn = []
behav = []
for i in range(len(dataset.subjects)):
    c = load_dynacomp_fc(dataset.subjects[i], session='func1', metric='pc', msdl=True)
    conn.append(c[i, j])
    b = behav_data[i]['postRT'] - behav_data[i]['preRT']
    behav.append(b)

sns.jointplot(np.array(conn), np.array(behav), kind='kde')
sns.axlabel('Connectivity', 'Behavior')

