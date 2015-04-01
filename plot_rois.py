# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 09:16:38 2015

@author: mr243268
"""
import os
from loader import load_dynacomp, set_figure_base_dir
from nilearn.plotting import plot_roi

dataset = load_dynacomp()

FIG_DIR = set_figure_base_dir('rois')

for i in range(len(dataset.subjects)):
    for k in sorted(dataset.rois[i].keys()):
        output_file = os.path.join(FIG_DIR, k)
        plot_roi(dataset.rois[i][k], title=k, output_file=output_file)
    break