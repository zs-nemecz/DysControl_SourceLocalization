# Collect source activity from all subjects and subtract right hemi data from left hemi data. Run t-test with cluster stats.

# Steps
# 1. Set data and out folder
# 2. Determine parameters: method, time window, number of permutations, significance level
# 3. Add subjects to be analyzed
# 4. Create source spaces: left and right hemisphere
# 5. Compute connectivity per hemisphere (this sh/ could have been done previously, outside the loop)
# 6. Read subject data for given method
# 7. Morph data to FSAverage_sym and create cross-hemi data
# 8. Normalized subtraction
# 9. Divide data into separate hemispheres
# 10. T-test with clustering
# 11. Save data as pickle file

import os.path as op
import pickle
import logging
import time
import numpy as np
from scipy import stats as stats
import mne
from mne.datasets import fetch_fsaverage
from mne.stats import (spatio_temporal_cluster_1samp_test)

# Logging settings
task = 'natural_reading'
logfile = op.join('..', '..', 'results', task , 'sym_src_t_test_clu.txt')
logging.basicConfig(filename=logfile, level=logging.INFO, filemode='a')
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %I:%M:%S %p')
logging.info('--------------------------------------------------------------------------------------------------------')

# 1. Set data and out folder
data_folder = op.join('..', '..', 'data')
source_folder = op.join('..', '..', 'source_activity', task)
out_folder = op.join('..', '..', 'results', task)
# Template subject file directory
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# 2. Determine parameters: method, time window, number of permutations, significance level
method = 'MNE'
## Select preferred time window
time_windows = [(0.14, 0.16), (0.175, 0.225), (0.245, 0.295)]
subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257',
            '103183', '862169', '284297', '643438', '048298', '414822', '638838', '390744', '930517',
            '093925', '213103', '331536', '205515', '135230', '320268', '319897', '321319', '303786',
            '822816', '163753', '667207', '424174', '612370', '528213', '009833', '927179', '515155',
            '366394', '133288']
n_subjects = len(subjects)
p_threshold = 0.05
n_permutations = 1024

# 4. Create source spaces: left and right hemisphere
# Use FSAverage_Sym - Template subject with registration between left and right hemi
src_fname = op.join('..', '..', 'fsaverage_sym', 'bem', 'fsaverage_sym_ico5-src.fif')
lh_src = mne.read_source_spaces(src_fname, verbose=True)
rh_src = lh_src.copy()
lh_src.pop()
rh_src.pop(0)
src = {'left_hemi':lh_src, 'right_hemi':rh_src}

logging.info('**** Computing connectivity *****')
info = {'SRC': src['left_hemi']}
logging.info(info)
# 5. Compute connectivity per hemisphere (this sh/ could have been done previously, outside the loop)
connectivity = mne.spatial_src_adjacency(src['left_hemi'], verbose=True)

for tmin, tmax in time_windows:
    time_label = str(int(tmin * 1000)) + '-' + str(int(tmax * 1000)) + 'ms'
    info = {'time-window':time_label,'p threshold for t-test':p_threshold, '# permutations':n_permutations,
            'Number of subjects':n_subjects}
    logging.info(info)

    start_time = time.time()
    logging.info('Starting time')
    logging.info(time.strftime("%H:%M:%S", time.gmtime(start_time)))
    print(time.strftime("%H:%M:%S", time.gmtime(start_time)))

    logging.info('.....................................{}.....................................'.format(method))
    ## Dummy variables for arrays built in the for loop
    stc_data = None
    stc_xhemi_data = None
    tstep = None
    X = None
    for subject in subjects:
        # 6. Read subject data for given method & time-window
        stc_file = op.join(source_folder, method, subject)
        stc = mne.read_source_estimate(stc_file, 'fsaverage')
        stc.crop(tmin, tmax)
        tstep = stc.tstep

        # 7. Morph data to FSAverage_sym and create cross-hemi data
        ## Morph to fsaverage_sym
        stc = mne.compute_source_morph(stc, 'fsaverage', 'fsaverage_sym', smooth=5,
                                       warn=False,
                                       subjects_dir=subjects_dir).apply(stc)
        # Compute a morph-matrix mapping the right to the left hemisphere,
        # and vice-versa.
        morph = mne.compute_source_morph(stc, 'fsaverage_sym', 'fsaverage_sym',
                                        spacing=stc.vertices, warn=False,
                                        subjects_dir=subjects_dir, xhemi=True,
                                        verbose='error')  # creating morph map
        stc_xhemi = morph.apply(stc)

        if np.all(stc_data) == None:
            stc_data = stc.data
            stc_xhemi_data = stc_xhemi.data
        else:
            stc_data = np.dstack((stc_data, stc.data))
            stc_xhemi_data = np.dstack((stc_xhemi_data, stc_xhemi.data))
    logging.info('STC data shape: {}'.format(stc_data.shape))
    logging.info('Xhemi data shape: {}'.format(stc_xhemi_data.shape))

    # 8. Normalized subtraction
    X = (stc_data[:, :, :] - stc_xhemi_data[:, :, :]) / (stc_data[:, :, :] + stc_xhemi_data[:, :, :])

    # 9. Divide data into separate hemispheres
    shape = X.shape
    single_hemi = int(shape[0]/2)
    l_hemi = X[:single_hemi, :, :] # take only the first halt, i.e. left hemisphere
    r_hemi = X[single_hemi:, :, :] # take second half, only right hemi
    logging.info('Subtracted left hemi data shape: {}'.format(l_hemi.shape))
    logging.info('Subtracted right hemi data shape: {}'.format(r_hemi.shape))

    # 10. T-test with clustering
    #    Data needs to be a multi-dimensional array of shape
    #    samples (subjects) x time x space, so we permute dimensions
    hemi_data = np.transpose(l_hemi, [2, 1, 0]) # reshape data for the test

    # Compute t-treshold based on p val and n subjects
    t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
    logging.info('----Clustering----')
    T_obs, clusters, cluster_p_values, H0 = clu = \
        spatio_temporal_cluster_1samp_test(hemi_data, adjacency=connectivity, n_permutations=n_permutations, n_jobs=4,
                                           threshold=t_threshold, buffer_size=None,
                                           verbose=True)

    # Some logging
    end_time = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(end_time)))
    elapsed_time = end_time - start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    logging.info('Elapsed since start:')
    logging.info(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    # 11. Log results and save data as pickle file
    good_cluster_inds = np.where(cluster_p_values < p_threshold)[0]
    ok_cluster_inds = np.where(cluster_p_values < 0.1)[0]
    pickle_fname = op.join(out_folder, 'left_hemi', method, 'cluster_timecourse', 'sym_src_normalized_contrast_' + time_label + '_nperm-' + str(int(n_permutations))+'_all_clu.pkl')
    logging.info('Clusters with p<0.05: {}'.format(good_cluster_inds))
    logging.info('Clusters with p<0.1: {}'.format(ok_cluster_inds))
    logging.info('Saving all_clu.pkl file to {}'.format(pickle_fname))
    with open(pickle_fname, 'wb') as f:
        pickle.dump(clu, f)

