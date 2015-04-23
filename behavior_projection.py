# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:45:16 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
import nibabel as nib
from loader import load_dynacomp, load_msdl_names_and_coords,\
                   load_dynacomp_fc, load_roi_names_and_coords,\
                   set_figure_base_dir                   
 
from nilearn.image import concat_imgs, mean_img
from nilearn.plotting import plot_roi, plot_stat_map, plot_img
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, Isomap
import matplotlib.pyplot as plt

msdl = False
dataset = load_dynacomp()
roi_names, roi_coords  = load_roi_names_and_coords(dataset.subjects[0])
if msdl:
    roi_names, roi_coords  = load_msdl_names_and_coords()

ind = np.tril_indices(len(roi_names), k=-1)

x = []
for subject_id in dataset.subjects:
    c = load_dynacomp_fc(subject_id=subject_id, session='func2', metric='gl',
                         msdl=msdl, preprocessing_folder='pipeline_2')
    x.append(c[ind])
x = np.array(x)

# Label vector y
groups = ['avn', 'v', 'av']
y = np.zeros(len(dataset.subjects))
for i, group in enumerate(['v', 'av']):
    y[dataset.group_indices[group]] = i + 1
    

pca = PCA(n_components=2)
pcx = pca.fit_transform(x)

mds = MDS(n_jobs=2, random_state=42)
px = mds.fit_transform(x)


groups = ['avn', 'v', 'av']
colors = ['r', 'g', 'b']
plt.figure()
for i, group in enumerate(groups):
    plt.scatter(px[dataset.group_indices[group], 0],
                px[dataset.group_indices[group], 1],
                s=80, c=colors[i])
plt.legend(groups, loc='lower right')
plt.grid(which='both')

p = px

# pairwise projection
for i in range(2):
    for j in range(i+1, 3):
        gr_i = dataset.group_indices[groups[i]]
        gr_j = dataset.group_indices[groups[j]]
        Xp = np.vstack((x[gr_i, :], x[gr_j, :]))
        mds = MDS(n_jobs=2, random_state=42)
        mds = Isomap(n_components=2)
        px = mds.fit_transform(Xp)
        plt.figure()
        plt.scatter(px[:len(gr_i), 0], px[:len(gr_i), 1], s=80, c='r')
        plt.scatter(px[len(gr_i) + 1:, 0], px[len(gr_i) + 1:, 1], s=80, c='b')
        plt.legend([groups[i], groups[j]], loc='lower right')
        plt.grid(which='both')


print p
plt.figure()
for k in range(len(px)):
    plt.text(p[k, 0], p[k, 1], s=dataset.subjects[k])
plt.xlim([np.min(p[:, 0]), np.max(p[:, 0])])
plt.ylim([np.min(p[:, 1]), np.max(p[:, 1])])
plt.title('subjects')