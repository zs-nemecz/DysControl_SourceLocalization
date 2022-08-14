# Create noise covariance matrices from fixed gaze data epochs from all conditions
# Steps:
# 1. Setup, add path and subject names
# 2. Read preprocessed EEG epochs
# 3. Create regularized and non-regularized noise covariance matrices
# 4. Write matrices to files

# 1. Setup, add path and subject names
import os.path as op
import mne

task = 'fixed_gaze'
data_folder = op.join('..', '..', 'data', task, 'preproc_EEG')
conditions = ['normal', 'phase_rand','armenian']
out_folder = op.join('..', '..', 'noise_cov', task, 'all')
subjects = ['842608', '587631', '217720', '181585', '059694', '394107', '356349', '044846', '050120', '269257', '103183',
            '862169', '284297', '643438', '048298', '414822', '638838', '390744', '930517', '093925', '213103', '331536',
            '205515', '135230', '320268', '319897', '321319', '303786', '822816', '163753', '667207', '424174', '612370',
            '528213', '009833', '927179', '515155', '366394', '133288']
print('Number of subjects included: ', len(subjects))
for subject in subjects:
    epochs = []

    # 2. Read preprocessed EEG epochs from all conditions per subject
    for condition in conditions:
        print('************************\n{}\n************************'.format(condition))
        data_file = op.join(data_folder, subject, 'nat_ica', 'erp_eeg_' + condition +'_v2_avgref.set')
        epochs.append(mne.io.read_epochs_eeglab(data_file))

# 3. Create regularized noise covariance matrices
    print('\nComputing regularized cov-mat for subject ', subject)
    reg_noise_cov_baseline = mne.compute_covariance(epochs, tmax=0., method='auto', rank=None, verbose=True) #regularized

# 4. Write matrices to files
    print('\nSaving matrices for subject ', subject)
    reg_noise_cov_file = op.join(out_folder, subject+'_reg_noise-cov.fif')
    mne.write_cov(reg_noise_cov_file,reg_noise_cov_baseline)
    print('Subject {} Done.'.format(subject))
    print('==========================================================================================================================================================')
    print(len(epochs))
