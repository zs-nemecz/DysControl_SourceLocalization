# Source localization with template anatomy and standard electrode locations

# Steps:
# 1. Setup and add path and subject names
# 2. Read previously created forward solution (see create_template_fwd.py)
# 3. Read previously created noise covariance matrix
# 4. Read preprocessed EEG evoked file (1 per subject)
# 5. Make inverse operator
# 6. Compute inverse solution


# 1. Setup and add path and subject names
import os.path as op
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse
import logging
import time

log_fname=op.join('..', '..', 'data', 'natural_reading', 'source_localization_log.txt')
logging.basicConfig(filename=log_fname,
                    level=logging.INFO, filemode='a')
logging.info('--------------------------------------------------------------------------------------------------------')

data_folder = op.join('..', '..', 'data')
task_folder = 'natural_reading'
out_folder = op.join('..', '..', 'source_activity', task_folder)
noise_cov_folder = op.join('..', '..', 'noise_cov', task_folder)
subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257',
            '103183', '862169', '284297', '643438', '048298', '414822', '638838', '390744', '930517',
            '093925', '213103', '331536', '205515', '135230', '320268', '319897', '321319', '303786',
            '822816', '163753', '667207', '424174', '612370', '528213', '009833', '927179', '515155',
            '366394', '133288']
inverse_solvers = ['MNE']
n_subjects = len(subjects)
loose = 'auto'
depth = 0.8
snr = 3.
lambda2 = 1. / snr ** 2

info = {'lambda':lambda2,'loose parameter (orientation)':loose, 'depth':depth,
        'Number of subjects':n_subjects}
logging.info(info)
start_time = time.time()
logging.info('Starting time')
logging.info(time.strftime("%H:%M:%S", time.gmtime(start_time)))
print(time.strftime("%H:%M:%S", time.gmtime(start_time)))

# 2. Read previously created forward solution (see create_template_fwd.py)
fwd_fname = op.join('..', '..', 'template_fwd', 'template-fwd.fif')
fwd = mne.read_forward_solution(fwd_fname)

for subject in subjects:
    logging.info(subject)
    # 3. Read previously created noise covariance matrix
    noise_cov_file = op.join(noise_cov_folder, subject + '_reg_noise-cov.fif')
    noise_cov = mne.read_cov(noise_cov_file)

    # 4. Read evoked file
    evoked_file = op.join('..', '..', 'evoked', task_folder, subject + '-ave.fif')
    evoked = mne.read_evokeds(evoked_file, condition=0, baseline=None, proj=False)

    # 5. Make inverse operator
    print('\nCreating inverse operator for subject ', subject)
    info = evoked.info
    inverse_operator = make_inverse_operator(info, fwd, noise_cov, loose=loose, depth=depth)
    del noise_cov

    # 6. Compute inverse solution
    for method in inverse_solvers:
        logging.info(method)
        print('\nComputing inverse solutions with method {} for subject {}'.format(method, subject))

        stc, residual = apply_inverse(evoked, inverse_operator, lambda2, method=method, pick_ori=None, return_residual=True, verbose=True)
        res_file = op.join(out_folder, method, 'residual', subject + '-ave.fif')
        residual.save(res_file)
        del residual

        stc_file = op.join(out_folder, method, subject)
        print('\nSaving inverse solution with method {} for subject {} \n to file: {}'.format(method, subject, stc_file))
        logging.info('\nSaving inverse solution with method {} for subject {} \n to file: {}'.format(method, subject, stc_file))
        stc.save(stc_file)
        del stc
    print('=========================================================================================================================================')

end_time = time.time()
print(time.strftime("%H:%M:%S", time.gmtime(end_time)))
elapsed_time = end_time - start_time
print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
logging.info('Elapsed since start:')
logging.info(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))