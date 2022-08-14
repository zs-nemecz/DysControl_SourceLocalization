# Read epoch data, make noise covariance matrices and create evoked files fro averaged epochs

# Note: ET regressed epoch data are in .mat matrices, without channel and event info
# .mat files can't be read into MNE automatically

# Steps
# 1. read both 'old' preprocessed epochs and 'new' .mat files, from which ET data was regressed out
# 2. average new epochs to get evoked response
# 3. create noise covariance matrices
# 4. save matrices and evoked file

import os.path as op
import mne
import numpy as np
import scipy.io as scio

# set paths, subjects, and file names
task = 'natural_reading'
data_folder = op.join('..', '..', 'data', task)
out_folder = '..'
old_data_folder = op.join(data_folder, 'preproc_EEG')
new_data_folder = 'REGROUT_t_avgref'
old_fname = 'fra_eeg_avgref.set'
new_fname = 'Res.mat'
subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257',
            '103183', '862169', '284297', '643438', '048298', '414822', '638838', '390744', '930517',
            '093925', '213103', '331536', '205515', '135230', '320268', '319897', '321319', '303786',
            '822816', '163753', '667207', '424174', '612370', '528213', '009833', '927179', '515155',
            '366394', '133288']

for subject in subjects:
    # 1. Read epochs
    # set epoch file name based on subject ID
    old_data_file = op.join(old_data_folder, subject, old_fname)
    new_data_file = op.join(data_folder, new_data_folder, subject, new_fname)

    # first read old epochs, which come with event info
    old_epochs = mne.io.read_epochs_eeglab(old_data_file)
    # new epochs don't have event info, so we use old events as 'dummy' events
    mat = scio.loadmat(new_data_file)
    data = np.transpose(mat['Res'], [2, 0, 1])
    events = old_epochs.events[:data.shape[0],:]
    new_epochs = old_epochs.copy()
    new_epochs._data = data
    new_epochs.events = events
    new_epochs.set_eeg_reference(projection=True)  # needed for inverse modeling

    # 2. Average the new epoch file to get evoked responses
    new_evoked = new_epochs.average().pick('eeg')

    # 3. Create regularized and non-regularized noise covariance matrices
    print('\nComputing non-regularized cov-mat for subject ', subject)
    noise_cov = mne.compute_covariance(new_epochs, tmin=-0.05,tmax=0.)  # default method: 'empirical'
    print('\nComputing regularized cov-mat for subject ', subject)
    reg_noise_cov = mne.compute_covariance(new_epochs, tmin=-0.05, tmax=0., method='auto', rank=None,
                                           verbose=True)  # regularized
    # 4. Save
    # write matrices to files
    print('\nSaving matrices for subject ', subject)
    noise_cov_file = op.join(out_folder, 'noise_cov', task, subject + '_noise-cov.fif')
    reg_noise_cov_file = op.join(out_folder, 'noise_cov', task, subject + '_reg_noise-cov.fif')
    mne.write_cov(noise_cov_file, noise_cov)
    mne.write_cov(reg_noise_cov_file, reg_noise_cov)
    print('Done.')
    print('===========================================================================================================')
    # finally, save the new evoked file
    evoked_fname = op.join(out_folder, 'evoked', task,  subject + '-ave.fif')
    new_evoked.save(evoked_fname)