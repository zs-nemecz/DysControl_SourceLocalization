import os.path as op
import mne

task = 'fixed_gaze'
data_folder = op.join('..', '..', 'data', task)
out_folder = '..'
conditions = ['armenian', 'normal', 'phase_rand']

subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257', '103183', '862169',
            '181585', '284297', '643438', '048298', '414822', '638838', '390744', '930517', '093925', '213103', '331536',
            '205515', '135230', '320268', '319897', '321319', '303786', '822816', '163753', '667207', '424174', '612370',
            '528213', '009833', '927179', '515155', '366394', '133288']

for subject in subjects:
    for condition in conditions:
        # Read preprocessed EEG epochs
        data_file = op.join(data_folder, subject, 'nat_ica', 'erp_eeg_' + condition  + '_v2_avgref.set')
        epochs = mne.io.read_epochs_eeglab(data_file)
        epochs.set_eeg_reference(projection=True)  # needed for inverse modeling

        # Compute and save the evoked response
        evoked = epochs.average().pick('eeg')
        evoked_fname = op.join(out_folder, 'evoked', task, condition, subject + '-ave.fif')
        evoked.condition = condition
        evoked.save(evoked_fname)