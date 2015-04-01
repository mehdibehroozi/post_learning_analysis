# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 09:59:43 2015

@author: mr243268
"""

import os
from loader import load_dynacomp, set_data_base_dir, set_figure_base_dir
from nilearn.decomposition import CanICA
from nilearn.plotting import plot_stat_map
from nilearn.image import iter_img


dataset = load_dynacomp()
func = dataset.func1

n_components = 20
canica = CanICA(n_components=n_components, mask=dataset.mask,
                smoothing_fwhm=None, do_cca=True, threshold=3.,
                n_init=10, standardize=True, random_state=42,
                n_jobs=2, verbose=2)

CANICA_PATH = set_data_base_dir('Dynacomp/canica')
output_file = os.path.join(CANICA_PATH,
                           'canica_' + str(n_components) + '.nii.gz')

if not os.path.isfile(output_file):
    canica.fit(func)
    components_img = canica.masker_.inverse_transform(canica.components_)
    components_img.to_filename(output_file)
else:
    components_img = output_file

FIG_PATH = set_figure_base_dir('canica')

for i, cur_img in enumerate(iter_img(components_img)):
    output_file = os.path.join(FIG_PATH,
                               'canica_'+ str(n_components) +'_IC%d' % i )
    plot_stat_map(cur_img, display_mode="z", title="IC %d" % i, cut_coords=1,
                  colorbar=False, output_file=output_file)
