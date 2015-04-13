# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 13:45:16 2015

@author: mehdi.rahim@cea.fr
"""

import os
import numpy as np
import nibabel as nib
from loader import load_dynacomp, load_msdl_names_and_coords,\
                   load_dynacomp_fc, load_roi_names_and_coords, set_figure_base_dir
                   
 
from nilearn.image import concat_imgs, mean_img
from nilearn.plotting import plot_roi, plot_stat_map, plot_img

dataset = load_dynacomp()
roi_names, roi_coords  = load_roi_names_and_coords(dataset.subjects[0])
            

imgs = []
for key in sorted(dataset.rois[0].keys()):
    print key
    imgs.append(dataset.rois[0][key])


cimgs = []
for i, img in enumerate(imgs):
    m = nib.load(img)
    c = nib.Nifti1Image((i+1) * (len(imgs)+1) * m.get_data(), m.get_affine())
    cimgs.append(c)

img = mean_img(cimgs)

plot_roi(img)
#indice = 143