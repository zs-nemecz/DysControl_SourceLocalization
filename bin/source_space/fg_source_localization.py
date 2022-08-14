# Source localization with template anatomy and standard electrode locations

# Steps:
# 1. Setup and add path and subject names
# 2. Read noise covariance matrix from preprocessed resting state data
# 3. Read evoked EEG data from
# 4. Read previously created forward solution (see create_template_fwd.py)
# 5. Make inverse operator
# 6. Compute inverse solution
# 7. Save res_file and stc_file

# 1. Setup and add path and subject names
import os.path as op
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse

task = 'fixed_gaze'
conditions = ['armenian', 'normal', 'phase_rand']
noise_cov_folder = op.join('..', 'noise_cov', task, 'all')

subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257', '103183', '862169',
            '181585', '284297', '643438', '048298', '414822', '638838', '390744', '930517', '093925', '213103', '331536',
            '205515', '135230', '320268', '319897', '321319', '303786', '822816', '163753', '667207', '424174', '612370',
            '528213', '009833', '927179', '515155', '366394', '133288']


method = 'MNE'
loose = 'auto'
depth = 0.8
print('Inverse operator characteristics \nLoose: {} \nDepth: {}'.format(loose,depth))
snr = 3.
lambda2 = 1. / snr ** 2

# 2. Read previously created forward solution (see create_template_fwd.py)
fwd_fname = op.join('..', 'template_fwd', 'template-fwd.fif')
fwd = mne.read_forward_solution(fwd_fname)

for condition in conditions:
    evoked_dir = op.join('..', 'evoked', task, condition)
    out_folder = op.join('..', 'source_activity', task, condition)
    for subject in subjects:

        # 3. Read noise covariance matrix from preprocessed resting state data
        noise_cov = mne.read_cov(op.join(noise_cov_folder, subject + '_reg_noise-cov.fif'))

        # 4. Read evoked files
        evoked_file =  op.join(evoked_dir, subject + '_' + condition + '-ave.fif')
        evoked = mne.read_evokeds(evoked_file, condition = 0, baseline=None, proj=False)

        # 5. Make inverse operator
        print('\nCreating inverse operator for subject ', subject)
        info = evoked.info
        inverse_operator = make_inverse_operator(info, fwd, noise_cov, loose=loose, depth=depth) # IMPORTANT, default loose value is 0.2
        del noise_cov

        # 6. Compute inverse solution
        print('\nComputing inverse solutions with method {} for subject {}'.format(method, subject))
        stc, residual = apply_inverse(evoked, inverse_operator, lambda2, method=method, pick_ori=None, return_residual=True, verbose=True)

        # 7. Save res_file and stc_file
        res_file = op.join(out_folder, method, 'residual', subject + '-ave.fif')
        residual.save(res_file)
        del residual

        stc_file = op.join(out_folder, method, subject)
        print('\nSaving inverse solution with method {} for subject {} \n to file: {}'.format(method, subject, stc_file))
        stc.save(stc_file)
        del stc
        print('=========================================================================================================================================')
