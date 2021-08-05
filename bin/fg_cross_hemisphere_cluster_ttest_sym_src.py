# Collect source activity from all subjects and subtract right hemi data from left hemi data.
# Run t-test on cross-hemi contrast with cluster stats.
# Then subtract the armenian cross-hemi contrast the normal condition.

# Steps
# 1. Define clustering function
# 2. Set data and out folder
# 3. Determine parameters: method, time window, number of permutations, significance level
# 4. Add subjects to be analyzed
# 5. Define source space and compute connectivity for left hemisphere
# 6. Read subject data
# 7. Morph data to FSAverage_sym and create cross-hemi data
# 8. Normalized subtraction
# 9. Divide data into separate hemispheres
# 10. T-test with clustering on contrast data for each condition
# 11. Condition contrast

import os.path as op
import numpy as np
import mne
import pickle
from mne.datasets import fetch_fsaverage
from mne.stats import (spatio_temporal_cluster_1samp_test)
from scipy import stats as stats
import time
import logging

logfile = op.join('..', 'results', 'fixed_gaze', 'sym_src_t_test_clu.txt')
logging.basicConfig(filename=logfile, level=logging.INFO,
                    filemode='a')

# 1. Define clustering function
def ttest_clustering(X, p_threshold, n_subjects, n_permutations, connectivity, log=True):
    if log:
        cluster_start = time.time()

    # X needs to be a multi-dimensional array of shape
    # samples (subjects) x time x space, so we permute dimensions
    X = np.transpose(X, [2, 1, 0])

    # t-treshold based on p-treshold and n of subjects
    t_threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
    # Spatiotemporal clustering
    T_obs, clusters, cluster_p_values, H0 = clu = \
        spatio_temporal_cluster_1samp_test(X, adjacency=connectivity, n_permutations=n_permutations, n_jobs=1,
                                           threshold=t_threshold, buffer_size=None, verbose=True)
    #    Now select the clusters that are sig. at p < 0.05 (note that this value
    #    is multiple-comparisons corrected).
    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
    ok_cluster_inds = np.where(cluster_p_values < 0.1)[0]
    print('Good cluster inds:')
    print(good_cluster_inds)
    logging.info('Clusters with p<0.05: {}'.format(good_cluster_inds))
    logging.info('Clusters with p<0.1: {}'.format(ok_cluster_inds))

    if log:
        cluster_finish = time.time()
        elapsed = cluster_finish - cluster_start
        logging.info('Clustering finished in {}'. format(time.strftime("%H:%M:%S", time.gmtime(elapsed))))
    return clu

# 2. Set data and out folder
task='fixed_gaze'
data_folder = op.join('..', 'source_activity', task)

src_fname = op.join('..', 'fsaverage_sym', 'bem', 'fsaverage_sym_ico5-src.fif')
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)

# 3. Determine parameters: method, time window, number of permutations, significance level
conditions = ['normal', 'armenian', 'phase_rand']
contrast = ['normal', 'armenian']
method = 'MNE'
time_windows = [(0.175, 0.335), (0.335, 0.440)]

## cluster analysis
n_permutations = 1024
p_threshold = 0.05
cluster_p = 0.05

# 4. Add subjects to be analyzed
subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257', '103183',
            '862169', '284297', '643438', '048298', '414822', '638838', '390744', '930517', '093925', '213103',
            '331536', '205515', '135230', '320268', '319897', '321319', '303786', '822816', '163753', '667207',
            '424174', '612370', '528213', '009833', '927179', '515155', '366394', '133288']
n_subjects = len(subjects)
logging.info('Number of subjects included: {}'.format(n_subjects))

# 5. Define source space and compute connectivity for left hemisphere
#separate right and left hemi source space
lh_src = mne.read_source_spaces(src_fname, verbose=True)
rh_src = lh_src.copy()
lh_src.pop()
rh_src.pop(0)
src = {'left_hemi':lh_src, 'right_hemi':rh_src}

print('Computing connectivity.')
connectivity = mne.spatial_src_adjacency(src['left_hemi'], verbose=True)
logging.info('Adjacency computed for {}'.format(src['left_hemi']))

start_time = time.time()
logging.info(time.strftime("%H:%M:%S", time.gmtime(start_time)))

for tmin, tmax in time_windows:
    time_label = str(int(tmin * 1000)) + '-' + str(int(tmax * 1000)) + 'ms'
    info = {'time-window':time_label,'p threshold for t-test':p_threshold, '# permutations':n_permutations,
            'Number of subjects':n_subjects}
    logging.info(info)

    logging.info('.....................................{}.....................................'.format(method))
    out_folder = op.join('..', 'results', task, 'cross_hemi', 'left_hemi', method, 'cluster_timecourse')
    left_hemi_data = {}
    right_hemi_data = {}
    for condition in conditions:
        logging.info(condition)
        stc_data = None
        stc_xhemi_data = None

        for subject in subjects:
            # 6. Read subject data
            stc_file = op.join(data_folder, condition, method, subject)
            stc = mne.read_source_estimate(stc_file, 'fsaverage')
            stc.crop(tmin, tmax)
            tstep = stc.tstep

            # 7. Morph data to FSAverage_sym and create cross-hemi data
            # Morph to fsaverage_sym
            stc = mne.compute_source_morph(stc, subject_to='fsaverage_sym', smooth=5,
                                           warn=False,
                                           subjects_dir=subjects_dir).apply(stc)
            # Compute a morph-matrix mapping the right to the left hemisphere,
            # and vice-versa.
            morph = mne.compute_source_morph(stc, 'fsaverage_sym', 'fsaverage_sym',
                                             spacing=stc.vertices, warn=False,
                                             subjects_dir=subjects_dir, xhemi=True,
                                             verbose='error')  # creating morph map
            stc_xhemi = morph.apply(stc)
            if np.all(stc_data) is None:
                stc_data = stc.data
                stc_xhemi_data = stc_xhemi.data
            else:
                stc_data = np.dstack((stc_data, stc.data))
                stc_xhemi_data = np.dstack((stc_xhemi_data, stc_xhemi.data))
        logging.info('Collected {} stc data files from {} condition'.format(stc_data.shape[2], condition))
        # 8. Normalized subtraction
        X = (stc_data[:, :, :] - stc_xhemi_data[:, :, :]) / (stc_data[:, :, :] + stc_xhemi_data[:, :, :])

        # 9. Divide data into separate hemispheres
        shape = X.shape
        single_hemi = int(shape[0] / 2)
        l_hemi = X[:single_hemi, :, :]  # take only the first halt, i.e. left hemisphere
        r_hemi = X[single_hemi:, :, :]  # take second half, only right hemi
        logging.info('Subtracted left hemi data shape: {}'.format(l_hemi.shape))
        logging.info('Subtracted right hemi data shape: {}'.format(r_hemi.shape))
        left_hemi_data[condition] = l_hemi
        right_hemi_data[condition] = r_hemi

        # 10. T-test with clustering on contrast data for each condition
        pickle_fname = op.join(out_folder, 'sym_src_normalized_contrast_' + condition + '_' + time_label + '_nperm-' + str(int(n_permutations)) +'_all_clu.pkl')
        clu = ttest_clustering(l_hemi, p_threshold, n_subjects, n_permutations, connectivity)
        #pickle clu object and save stc with good clusters (p<0.05)
        with open(pickle_fname, 'wb') as f:
            pickle.dump(clu, f)

    logging.info('Collected cross hemi contrasts for {} conditions'.format(len(left_hemi_data)))
    logging.info([c for c in left_hemi_data])
    cross_hemi = {'left_hemi': left_hemi_data, 'right_hemi': right_hemi_data}

    # 11. Condition contrast
    print('Comparing {} with {} condition'.format(contrast[0], contrast[1]))
    logging.info('Contrast: {}'.format(contrast))
    C = cross_hemi['left_hemi'][contrast[0]] - cross_hemi['left_hemi'][contrast[1]]
    pickle_fname = op.join(out_folder, 'sym_src_normalized_contrast_' + contrast[0] + '_' + contrast[1] + time_label + '_nperm-' + str(int(n_permutations)) + '_all_clu.pkl')
    # Clustering
    clu = ttest_clustering(C, p_threshold, n_subjects, n_permutations, connectivity)
    with open(pickle_fname, 'wb') as f:
        pickle.dump(clu, f)

end_time = time.time()
logging.info(time.strftime("%H:%M:%S", time.gmtime(end_time)))
elapsed_time = end_time - start_time
logging.info('Elapsed time:')
logging.info(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
