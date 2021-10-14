# This script produces the figures showing the ROI time courses with standard deviation for the Fixed Gaze results

import os.path as op
import numpy as np
import mne
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as sp
import matplotlib.ticker as tck

# visualization and matplotlib settings
mne.viz.set_3d_backend('pyvista')
plt.rcParams.update({'figure.max_open_warning': 0})
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
matplotlib.rcParams.update(new_rc_params)
plt.rcParams['figure.dpi'] = 142
#plt.rcParams["figure.figsize"] = (18.0*cm,12.0*cm) 
font = {'family' : 'Arial',
        'weight' : 'bold',
        'size'   : 8}
plt.style.use('default')
alpha=0.3
cm = 1/2.54
cmap='RdBu_r'
lw=1.5

# define subjects (individual data needed for standard deviation)
subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257',
            '103183', '862169', '284297', '643438', '048298', '414822', '638838', '390744', '930517',
            '093925', '213103', '331536', '205515', '135230', '320268', '319897', '321319', '303786',
            '822816', '163753', '667207', '424174', '612370', '528213', '009833', '927179', '515155',
            '366394', '133288']
n_subjects = len(subjects)

# Function for extracting and plotting data from a label
def x_roi_timecourse_fig(stc, xstc, src, label, mode='mean', ax=None):
    '''Extract time course data from labels and visualize them.
    stc: stc of left hemisphere
    xstc: stc of right hemisphere
    label: name of ROI in the left hemisphere as in atlas, string
    mode: as in the MNE function extract_label_time_course
    ax: matplotlib axis'''

    # define colors for each hemi
    c_lh=(213/255.0, 94/255.0, 0)
    c_rh= (0, 114/255.0, 178/255.0)

    # use exponent for visualization
    exponent=10**13

    #extract labels
    roi_lh = mne.extract_label_time_course(stc, label, src, mode=mode)* exponent
    roi_rh = mne.extract_label_time_course(xstc, label, src, mode=mode)* exponent

    # plot
    ax.plot(1000 * stc.times, roi_lh.T, color=c_lh, label='left', linewidth=lw) #left hemi
    ax.fill_between(1000 * stc.times, y1=roi_lh.T[:,0]-(stc_sem[i]*exponent), y2=roi_lh.T[:,0]+(stc_sem[i]*exponent), color=c_lh, alpha=alpha, linewidth=0)

    ax.plot(1000 * stc.times, roi_rh.T, color=c_rh, label='right', linewidth=lw) #right hemi
    ax.fill_between(1000 * stc.times, y1=roi_rh.T[:,0]-(xhemi_sem[i]*exponent), y2=roi_rh.T[:,0]+(xhemi_sem[i]*exponent), color= c_rh, alpha=alpha, linewidth=0)
    ax.set_xticks(np.arange(stc.times[0]*1000, stc.times[-1]*1000, 50), minor= True)
    ax.set_xlim(xmin=0.0,xmax=350)
    ax.set_xticks([0,100,200,300])
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
    
    ax.tick_params(axis='both', which='major', labelsize=6, pad=0.5)
    ax.tick_params(axis='both', which='minor', labelsize=4, pad=0.5)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax

# 1. Select labels of interest
# read labels from atlas and create a dictionary with label.name:index
subjects_dir = op.join(mne.datasets.sample.data_path(), 'subjects')
mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=subjects_dir,
                                          verbose=True)
labels = mne.read_labels_from_annot('fsaverage', 'aparc_sub', subjects_dir=subjects_dir)
label_keys = {label.name: i for i,label in enumerate(labels)}

# create custom labels
# lateral occipital label (LO)
lat_occ_2 = labels[label_keys['lateraloccipital_2-lh']]
lat_occ_3 = labels[label_keys['lateraloccipital_3-lh']]
LO = lat_occ_2.copy()
LO.vertices = np.concatenate((lat_occ_2.vertices, lat_occ_3.vertices))
LO.values = np.concatenate((lat_occ_2.values, lat_occ_3.values))
LO.name = 'LO'

# middle ventral temporal label (MV)
it_6 = labels[label_keys['inferiortemporal_6-lh']]
l1,l2,l3,l4,l5 = it_6.split(5)
l31, l32 = l3.split(2)
l41, l42, l43 = l4.split(3)
p_it6 = it_6.copy()
p_it6.vertices = np.concatenate((l1.vertices,l2.vertices,l31.vertices,l41.vertices))
p_it6.values = np.concatenate((l1.values,l2.values,l31.values,l41.values))
p_it6.name = 'pIT6'

it_8 = labels[label_keys['inferiortemporal_8-lh']]
dorsal_it_8, ventral_it_8 = it_8.split(('dorsal', 'ventral'))
fus_4 = labels[label_keys['fusiform_4-lh']]
fus_3 = labels[label_keys['fusiform_3-lh']]
inf_temp_7 = labels[label_keys['inferiortemporal_7-lh']]
lat_occ_9 = labels[label_keys['lateraloccipital_9-lh']]

posterior_it_6, mid1_it_6, mid2_it_6, anterior_it_6 = it_6.split(4)
dorsal_lat_occ_9, ventral_lat_occ_9 = lat_occ_9.split(('dorsal', 'ventral'))
ventral_it_7, dorsal_it_7 = inf_temp_7.split(('ventral', 'dorsal'))
MV = inf_temp_7.copy()
MV.vertices = np.concatenate((ventral_it_8.vertices, fus_3.vertices, fus_4.vertices, ventral_it_7.vertices, ventral_lat_occ_9.vertices,p_it6.vertices))
MV.values = np.concatenate((ventral_it_8.values, fus_3.values, fus_4.values,ventral_it_7.values, ventral_lat_occ_9.values, p_it6.values))
MV.name = 'MV'


# inferior parietal label (IPT)
inf_par_3 = labels[label_keys['inferiorparietal_3-lh']]
inf_par_4 = labels[label_keys['inferiorparietal_4-lh']]

IP = MV.copy()
IP.vertices = np.concatenate((inf_par_3.vertices, inf_par_4.vertices))
IP.values = np.concatenate((inf_par_3.values, inf_par_4.values))
IP.name = 'IPT'

# temporo-parietal occipital label (TPO)
lat_occ_7 = labels[label_keys['lateraloccipital_7-lh']]
posterior_lat_occ_7, anterior_lat_occ_7 = lat_occ_7.split(2)
posterior_lat_occ_7, anterior_lat_occ_7 = lat_occ_7.split(('posterior', 'anterior'))
lat_occ_8 = labels[label_keys['lateraloccipital_8-lh']]

mid_temp_1 = labels[label_keys['middletemporal_1-lh']]
mid_temp_2 = labels[label_keys['middletemporal_2-lh']]

LP = MV.copy()
LP.vertices = np.concatenate((IP.vertices, anterior_lat_occ_7.vertices, lat_occ_8.vertices, dorsal_lat_occ_9.vertices, mid_temp_1.vertices, mid_temp_2.vertices, dorsal_it_8.vertices))
LP.values = np.concatenate((IP.values, anterior_lat_occ_7.values, lat_occ_8.values, dorsal_lat_occ_9.values, mid_temp_1.values, mid_temp_2.values, dorsal_it_8.values))
LP.name = 'TPO'

# posterior temporal label
temporal_LP = LP.copy()
temporal_LP.vertices = np.concatenate((mid_temp_1.vertices, mid_temp_2.vertices, dorsal_it_8.vertices))
temporal_LP.values = np.concatenate((mid_temp_1.values, mid_temp_2.values, dorsal_it_8.values))
temporal_LP.name = 'PT'

# anterior lateral occipital label
occipital_LP = temporal_LP.copy()
occipital_LP.vertices = np.concatenate((anterior_lat_occ_7.vertices, lat_occ_8.vertices, dorsal_lat_occ_9.vertices))
occipital_LP.values = np.concatenate((anterior_lat_occ_7.values, lat_occ_8.values, dorsal_lat_occ_9.values))
occipital_LP.name = 'LOa'

# posterior ventral occipito-temporal
fus_1 = labels[label_keys['fusiform_1-lh']]
fus_2 = labels[label_keys['fusiform_2-lh']]
lat_occ_10 = labels[label_keys['lateraloccipital_10-lh']]
lat_occ_11 = labels[label_keys['lateraloccipital_11-lh']]
PV = MV.copy()
PV.vertices = np.concatenate((fus_1.vertices, fus_2.vertices, lat_occ_10.vertices,lat_occ_11.vertices))
PV.values = np.concatenate((fus_1.values, fus_2.values,lat_occ_10.values,lat_occ_11.values))
PV.name = 'PV'

# define ROIs for visualization types (for mean ROI time course and mean lateralization time course)
rois = [LO, IP, temporal_LP, occipital_LP, PV, MV, LP]
ml_rois = [LO, PV, MV, LP]


# 2. Extract time courses for each condition

# define data folders
data_folder = op.join('..', '..', 'source_activity', 'fixed_gaze')
src_fname = op.join(mne.datasets.fetch_fsaverage(), 'bem', 'fsaverage-ico-5-src.fif')
src = mne.read_source_spaces(src_fname, verbose=True)
noise_cov = 'merged_nc'
method = 'MNE'

# normal condition, mean activity
condition='normal'
source_folder = op.join(data_folder, condition, noise_cov, method)
stc_file = op.join(data_folder, condition, noise_cov, method, condition + '_average')
norm_average_stc = mne.read_source_estimate(stc_file, 'fsaverage')

# morph from fsaverage to fsaverage_sym
normal_stc_xhemi = mne.compute_source_morph(norm_average_stc, 'fsaverage', 'fsaverage_sym', smooth=5,
                               warn=False,
                               subjects_dir=subjects_dir).apply(norm_average_stc)
# morph from left hemi to right, and vice versa
normal_stc_xhemi = mne.compute_source_morph(normal_stc_xhemi, 'fsaverage_sym', 'fsaverage_sym',
                                 spacing=normal_stc_xhemi.vertices, warn=False,
                                 subjects_dir=subjects_dir, xhemi=True,
                                 verbose='error').apply(normal_stc_xhemi)
# morph back the cross-hemi data to fsaverage space
# this step is needed for the labels to work (no labels for fsaverage_sym)
normal_stc_xhemi = mne.compute_source_morph(normal_stc_xhemi,'fsaverage_sym', 'fsaverage', smooth=5,
                               warn=False,
                               subjects_dir=subjects_dir).apply(normal_stc_xhemi)

# repeat morhing steps for all subjects, to obtain standard deviation of activity in sample
mode='mean'
norm_stc_data = []
norm_stc_xhemi_data = []
for subject in subjects:
    roi_data = []
    xroi_data = []
    stc_file = op.join(source_folder, subject)
    stc = mne.read_source_estimate(stc_file, 'fsaverage')
    stc.crop()
    tstep = stc.tstep
    ## Morph to fsaverage_sym
    stc_xhemi = mne.compute_source_morph(stc, 'fsaverage', 'fsaverage_sym', smooth=5,
                                   warn=False,
                                   subjects_dir=subjects_dir).apply(stc)
    # Compute a morph-matrix mapping the right to the left hemisphere,
    # and vice-versa.
    stc_xhemi = mne.compute_source_morph(stc_xhemi, 'fsaverage_sym', 'fsaverage_sym',
                                    spacing=stc_xhemi.vertices, warn=False,
                                    subjects_dir=subjects_dir, xhemi=True,
                                    verbose='error').apply(stc_xhemi)
    stc_xhemi = mne.compute_source_morph(stc_xhemi, 'fsaverage_sym', 'fsaverage', smooth=5,
                               warn=False,
                               subjects_dir=subjects_dir).apply(stc_xhemi)

    # extract time courses for each ROI for all subject
    for roi in rois:
        roi_data.append(mne.extract_label_time_course(stc, roi, src, mode=mode, verbose=False).squeeze())
        xroi_data.append(mne.extract_label_time_course(stc_xhemi, roi, src, mode=mode, verbose=False).squeeze())
    norm_stc_data.append(roi_data)
    norm_stc_xhemi_data.append(xroi_data)

# create array and calculate STD and SEM
norm_stc_data=np.array(norm_stc_data)
norm_stc_xhemi_data=np.array(norm_stc_xhemi_data)
stc_std = []
stc_sem = []
xhemi_std = []
xhemi_sem = []
for i in range(len(rois)):
    stc_std.append(np.std(norm_stc_data[:,i,:],axis=0))
    stc_sem.append(sp.sem(norm_stc_data[:,i,:],axis=0))
    
    xhemi_std.append(np.std(norm_stc_xhemi_data[:,i,:],axis=0))
    xhemi_sem.append(sp.sem(norm_stc_xhemi_data[:,i,:],axis=0))

# create figure
figure_roi = plt.figure(figsize=(18*cm,2.5*cm), dpi=142.12)
widths = [1,1,1,1,1,1]
heights = [1]
gs_rois = figure_roi.add_gridspec(ncols=6, nrows=1, width_ratios=widths,
                          height_ratios=heights, left=0.10, right=0.990, bottom=0.1)
gs_rois.update(wspace=0.5, hspace=0.5)


for i, label in enumerate(rois[:6]):
    axs1 = plt.subplot(gs_rois[i])
    x_roi_timecourse_fig(norm_average_stc, normal_stc_xhemi, src,  label, mode='mean', ax=axs1)#
    
    # exponent in offset text
    axs1.yaxis.offsetText.set_fontsize(6)

    axs1.set_ylabel('{} MSA\n(x e-13 a.u.)'.format(label.name), size=6, labelpad=0.6)
    if i == 5:
        plt.legend(fontsize=6, markerscale=0.1, prop={'size': 6}, frameon=False)


plt.show()
matplotlib.rcParams.update(new_rc_params)
figure_roi.savefig('fg_normal_rois_av_with_sem_1014.svg', figsize=(18*cm, 2.5*cm), dpi=600)

# armenian condition
condition= 'armenian'
method = 'MNE'
source_folder = op.join(data_folder, condition, noise_cov, method)
stc_file = op.join(data_folder, condition, noise_cov, method, condition + '_average')
arm_average_stc = mne.read_source_estimate(stc_file, 'fsaverage')

# morphing steps:
# 1. fsaverage -> fsaverage_sym
# 2. left to right and vice versa
# 3. fsaverage_sym -> fsaverage
arm_stc_xhemi = mne.compute_source_morph(arm_average_stc, 'fsaverage', 'fsaverage_sym', smooth=5,
                               warn=False,
                               subjects_dir=subjects_dir).apply(arm_average_stc)
arm_stc_xhemi = mne.compute_source_morph(arm_stc_xhemi, 'fsaverage_sym', 'fsaverage_sym',
                                 spacing=arm_stc_xhemi.vertices, warn=False,
                                 subjects_dir=subjects_dir, xhemi=True,
                                 verbose='error').apply(arm_stc_xhemi)
arm_stc_xhemi = mne.compute_source_morph(arm_stc_xhemi,'fsaverage_sym', 'fsaverage', smooth=5,
                               warn=False,
                               subjects_dir=subjects_dir).apply(arm_stc_xhemi)

# morph and extract for all subjects
mode='mean'
arm_stc_data = []
arm_xhemi_data = []
for subject in subjects:
    roi_data = []
    xroi_data = []
    stc_file = op.join(source_folder, subject)
    stc = mne.read_source_estimate(stc_file, 'fsaverage')
    stc.crop()
    tstep = stc.tstep
    ## Morph to fsaverage_sym
    stc_xhemi = mne.compute_source_morph(stc, 'fsaverage', 'fsaverage_sym', smooth=5,
                                   warn=False,
                                   subjects_dir=subjects_dir).apply(stc)
    # Compute a morph-matrix mapping the right to the left hemisphere,
    # and vice-versa.
    stc_xhemi = mne.compute_source_morph(stc_xhemi, 'fsaverage_sym', 'fsaverage_sym',
                                    spacing=stc_xhemi.vertices, warn=False,
                                    subjects_dir=subjects_dir, xhemi=True,
                                    verbose='error').apply(stc_xhemi)
    stc_xhemi = mne.compute_source_morph(stc_xhemi, 'fsaverage_sym', 'fsaverage', smooth=5,
                               warn=False,
                               subjects_dir=subjects_dir).apply(stc_xhemi)
    
    for roi in rois:
        roi_data.append(mne.extract_label_time_course(stc, roi, src, mode=mode, verbose=False).squeeze())
        xroi_data.append(mne.extract_label_time_course(stc_xhemi, roi, src, mode=mode, verbose=False).squeeze())
    arm_stc_data.append(roi_data)
    arm_xhemi_data.append(xroi_data)


# compute error
arm_stc_data=np.array(arm_stc_data)
arm_xhemi_data=np.array(arm_xhemi_data)
stc_std = []
stc_sem = []
xhemi_std = []
xhemi_sem = []
for i in range(len(rois)):
    stc_std.append(np.std(arm_stc_data[:,i,:],axis=0))
    stc_sem.append(sp.sem(arm_stc_data[:,i,:],axis=0))
    
    xhemi_std.append(np.std(arm_xhemi_data[:,i,:],axis=0))
    xhemi_sem.append(sp.sem(arm_xhemi_data[:,i,:],axis=0))

# create figures
figure_roi = plt.figure(figsize=(18*cm,2.5*cm), dpi=142.12)
widths = [1,1,1,1,1,1]
heights = [1]
gs_rois = figure_roi.add_gridspec(ncols=6, nrows=1, width_ratios=widths,
                          height_ratios=heights, left=0.10, right=0.99, bottom=0.1)
gs_rois.update(wspace=0.5, hspace=0.5)


for i, label in enumerate(rois[:6]):
    axs1 = plt.subplot(gs_rois[i])
    x_roi_timecourse_fig(arm_average_stc, arm_stc_xhemi, src,  label, mode='mean', ax=axs1)#

    axs1.yaxis.offsetText.set_fontsize(6)

    axs1.set_ylabel('{} MSA\n(x e-13 a.u.)'.format(label.name), size=6, labelpad=0.6)
    axs1.set_xlabel('Time (ms)', fontsize=7, labelpad=0.5)
    bottom, top = axs1.get_ylim()
    line_coord = (abs(bottom-top) / 100)
    axs1.plot(np.arange(175, 335, 1), np.full(160, line_coord + bottom), 'k', linewidth=lw)

    axs1.plot(np.arange(175, 220, 1), np.full(45, (line_coord*4) + bottom), color=(230 / 255.0, 159 / 255.0, 0),
                linewidth=lw)
    axs1.plot(np.arange(220, 290, 1), np.full(70, (line_coord*4) + bottom), color=(0, 158 / 255.0, 115 / 255.0),
                linewidth=lw)

plt.show()
matplotlib.rcParams.update(new_rc_params)
figure_roi.savefig('fg_armenian_rois_av_with_sem_1014.svg', figsize=(18*cm, 2.5*cm), dpi=600)

# normal - armenian difference
condition='normal'
stc_file = op.join(data_folder, condition, noise_cov, method, condition + '_average')
normal_average_stc = mne.read_source_estimate(stc_file, 'fsaverage')

condition='armenian'
stc_file = op.join(data_folder, condition, noise_cov, method, condition + '_average')
armenian_average_stc = mne.read_source_estimate(stc_file, 'fsaverage')

#norm xhemi
norm_stc_xhemi = mne.compute_source_morph(normal_average_stc, 'fsaverage', 'fsaverage_sym', smooth=5,
                               warn=False,
                               subjects_dir=subjects_dir).apply(normal_average_stc)

norm_stc_xhemi = mne.compute_source_morph(norm_stc_xhemi, 'fsaverage_sym', 'fsaverage_sym',
                                 spacing=norm_stc_xhemi.vertices, warn=False,
                                 subjects_dir=subjects_dir, xhemi=True,
                                 verbose='error').apply(norm_stc_xhemi)
norm_stc_xhemi = mne.compute_source_morph(norm_stc_xhemi, 'fsaverage_sym', 'fsaverage',smooth=5,
                               warn=False,
                               subjects_dir=subjects_dir).apply(norm_stc_xhemi)
# lateralization of normal condition (average)
norm_stc_xhemi.data = (normal_average_stc.data - norm_stc_xhemi.data)/(normal_average_stc.data + norm_stc_xhemi.data)

#arm xhemi
arm_stc_xhemi = mne.compute_source_morph(armenian_average_stc, 'fsaverage', 'fsaverage_sym', smooth=5,
                               warn=False,
                               subjects_dir=subjects_dir).apply(armenian_average_stc)

arm_stc_xhemi = mne.compute_source_morph(arm_stc_xhemi, 'fsaverage_sym', 'fsaverage_sym',
                                 spacing=norm_stc_xhemi.vertices, warn=False,
                                 subjects_dir=subjects_dir, xhemi=True,
                                 verbose='error').apply(arm_stc_xhemi)
arm_stc_xhemi = mne.compute_source_morph(arm_stc_xhemi, 'fsaverage_sym', 'fsaverage', smooth=5,
                               warn=False,
                               subjects_dir=subjects_dir).apply(arm_stc_xhemi)

# lateralization of armenian condition (average)
arm_stc_xhemi.data = (armenian_average_stc.data - arm_stc_xhemi.data)/(armenian_average_stc.data + arm_stc_xhemi.data)

# # lateralization of the two conditions (arrays with data for each subject)
armenian_ml = (arm_stc_data - arm_xhemi_data)/(arm_stc_data + arm_xhemi_data)
normal_ml = (norm_stc_data - norm_stc_xhemi_data)/(norm_stc_data + norm_stc_xhemi_data)

# calculate error for lateralization
stc_std = []
stc_sem = []
xhemi_std = []
xhemi_sem = []
for i in [0,3,4,5]:
    stc_std.append(np.std(normal_ml[:,i,:],axis=0))
    stc_sem.append(sp.sem(normal_ml[:,i,:],axis=0))
    
    xhemi_std.append(np.std(armenian_ml[:,i,:],axis=0))
    xhemi_sem.append(sp.sem(armenian_ml[:,i,:],axis=0))



def cond_x_roi_timecourse_fig(stc, xstc, src, label, mode='mean', ax=None, lc='orange'):
    '''Extract time course data from labels and visualize them.'''
    c_lh=(213/255.0, 94/255.0, 0)
    c_rh= (0, 114/255.0, 178/255.0)
    lh_label = label
    roi_lh = mne.extract_label_time_course(stc, lh_label, src, mode=mode)
    roi_rh = mne.extract_label_time_course(xstc, lh_label, src, mode=mode)
    ax.plot(1000 * stc.times, roi_lh.T, color=c_lh, label='Word') #left hemi
    ax.plot(1000 * stc.times, roi_rh.T, color=c_rh, label='FF') #right hemi
    ax.fill_between(1000 * stc.times, y1=roi_lh.T[:,0]-(stc_sem[i]), y2=roi_lh.T[:,0]+(stc_sem[i]), color=c_lh, alpha=alpha, linewidth=0)
    ax.fill_between(1000 * stc.times, y1=roi_rh.T[:,0]-(xhemi_sem[i]), y2=roi_rh.T[:,0]+(xhemi_sem[i]), color= c_rh, alpha=alpha, linewidth=0)
    ax.set_xticks(np.arange(stc.times[0]*1000, stc.times[-1]*1000, 50), minor= True)
    ax.set_xlim(xmin=0,xmax=350)
    ax.set_xticks([0,100,200,300])
    ax.set_ylim(ymin=-0.3,ymax=0.3)
    ax.tick_params(axis='both', which='major', labelsize=6, pad=0.5)
    ax.tick_params(axis='both', which='minor', labelsize=4, pad=0.5)
    
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    return ax


# plot lateralization

figure_roi = plt.figure(figsize=(6.5*cm,5.5*cm), dpi=142.12)
widths = [1,1]
heights = [1,1]
gs_rois = figure_roi.add_gridspec(ncols=2, nrows=2, width_ratios=widths,
                          height_ratios=heights, left=0.1, right=0.99, bottom=0.1)
gs_rois.update(wspace=0.5, hspace=0.5)

for i, label in enumerate(ml_rois):
    axs1 = plt.subplot(gs_rois[i])
    cond_x_roi_timecourse_fig(norm_stc_xhemi, arm_stc_xhemi, src,  label, mode='mean', ax=axs1)#

    axs1.yaxis.offsetText.set_fontsize(6)
    axs1.set_ylabel('{} ML'.format(label.name), size=6, labelpad=0.6)

    if i > 1:
        axs1.set_xlabel('Time (ms)', fontsize=7, labelpad=0.5)
        
    if i == 1:
        plt.legend(fontsize=6, markerscale=0.1, prop={'size': 6}, frameon=False)
axs = figure_roi.get_axes()
axs[1].legend(fontsize=6, markerscale=0.1, prop={'size': 5}, frameon=False)
axs[2].plot(np.arange(175,335,1),np.full(160, (0.6/20)-0.3),  'k', linewidth=lw)

axs[2].plot(np.arange(175,220,1),np.full(45, (0.6/20)-0.28),  color = (230/255.0, 159/255.0, 0), linewidth=lw)
axs[2].plot(np.arange(220,290,1),np.full(70, (0.6/20)-0.28), color = (0, 158/255.0, 115/255.0), linewidth=lw)

axs[3].plot(np.arange(175,335,1),np.full(160, (0.6/20)-0.3),  'k', linewidth=lw)

axs[3].plot(np.arange(175,220,1),np.full(45, (0.6/20)-0.28), color = (230/255.0, 159/255.0, 0), linewidth=lw)
axs[3].plot(np.arange(220,290,1),np.full(70, (0.6/20)-0.28), color = (0, 158/255.0, 115/255.0), linewidth=lw)

plt.show()
matplotlib.rcParams.update(new_rc_params)
figure_roi.savefig('fg_normal-armenian_rois_av_with_sem_1014.svg', figsize=(18*cm, 2.5*cm), dpi=600)




