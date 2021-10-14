
import os.path as op

import numpy as np
import mne

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import scipy.stats as sp
import matplotlib.ticker as tck


mne.viz.set_3d_backend('pyvista')
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
subjects_dir = mne.datasets.sample.data_path() + '/subjects'

# Function for extracting and plotting data from a label
def x_roi_timecourse_fig(stc, xstc, src, label, mode='mean', ax=None, exponent=1):
    '''Extract time course data from labels and visualize them.'''
    c_lh=(213/255.0, 94/255.0, 0)
    c_rh= (0, 114/255.0, 178/255.0)
    lh_label = label
    roi_lh = mne.extract_label_time_course(stc, lh_label, src, mode=mode)*exponent
    roi_rh = mne.extract_label_time_course(xstc, lh_label, src, mode=mode)*exponent
    print(roi_lh.T.shape)
    ax.plot(1000 * stc.times, roi_lh.T, color=c_lh, label='left', linewidth=lw) #left hemi
    ax.fill_between(1000 * stc.times, y1=roi_lh.T[:,0]-(stc_sem[i]*exponent), y2=roi_lh.T[:,0]+(stc_sem[i]*exponent), color=c_lh, alpha=alpha, linewidth=0)
    ax.plot(1000 * stc.times, roi_rh.T, color=c_rh, label='right', linewidth=lw) #right hemi
    ax.fill_between(1000 * stc.times, y1=roi_rh.T[:,0]-(xhemi_sem[i]*exponent), y2=roi_rh.T[:,0]+(xhemi_sem[i]*exponent), color= c_rh, alpha=alpha, linewidth=0)
    ax.set_xticks(np.arange(stc.times[0]*1000, stc.times[-1]*1000, 50), minor= True)
    ax.set_xlim(xmin=0,xmax=350)
    ax.set_xticks([0, 100, 200, 300])
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator())

    ax.tick_params(axis='both', which='major', labelsize=6, pad=0.5)
    ax.tick_params(axis='both', which='minor', labelsize=4, pad=0.5)

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax

# 1. Select labels of interest
# read labels from atlas and create a dictionary with label.name:index
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, verbose=True)

mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=subjects_dir, verbose=True)

labels = mne.read_labels_from_annot('fsaverage', 'aparc_sub')
label_keys = {label.name: i for i,label in enumerate(labels)}

hcp_labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'lh', subjects_dir=subjects_dir)

# lateral occipital (LO)
lat_occ_2 = labels[label_keys['lateraloccipital_2-lh']]
lat_occ_3 = labels[label_keys['lateraloccipital_3-lh']]
LO = lat_occ_2.copy()
LO.vertices = np.concatenate((lat_occ_2.vertices, lat_occ_3.vertices))
LO.values = np.concatenate((lat_occ_2.values, lat_occ_3.values))
LO.name = 'LO'

# superior premotor (SPM)
SPM = LO.copy()
l_fef = hcp_labels[62]
l_55b = hcp_labels[29]
hcp_label_keys = {hcp_label.name: i for i,hcp_label in enumerate(hcp_labels)}
SPM.vertices = np.concatenate((l_fef.vertices,l_55b.vertices))
SPM.values = np.concatenate((l_fef.values,l_55b.values))
SPM.name = 'SPM'

# inferior premotor (IPM)
IPM = SPM.copy()
L_6r = hcp_labels[hcp_label_keys['L_6r_ROI-lh']]
L_6v = hcp_labels[hcp_label_keys['L_6v_ROI-lh']]
L_IFJa = hcp_labels[hcp_label_keys['L_IFJa_ROI-lh']]
L_IFJp = hcp_labels[hcp_label_keys['L_IFJp_ROI-lh']]
dorsal_L_6v, ventral_L_6v =  L_6v.split(('dorsal', 'ventral'))
IPM.vertices = np.concatenate((L_6r.vertices, L_6v.vertices, L_IFJa.vertices, L_IFJp.vertices))
IPM.values = np.concatenate((L_6r.values, L_6v.values, L_IFJa.values,L_IFJp.values))
IPM.name = 'IPM'

# middle ventral temporal
it_6 = labels[label_keys['inferiortemporal_6-lh']]
l1,l2,l3,l4,l5 = it_6.split(5)
l31, l32 = l3.split(2)
l41, l42, l43 = l4.split(3)
p_it6 = it_6.copy()
p_it6.vertices = np.concatenate((l1.vertices,l2.vertices,l31.vertices,l41.vertices))
p_it6.values = np.concatenate((l1.values,l2.values,l31.values,l41.values))
p_it6.name = 'pIT6'

a_it6 = it_6.copy()
a_it6.vertices = np.concatenate((l32.vertices,l42.vertices,l43.vertices,l5.vertices))
a_it6.values = np.concatenate((l32.values,l42.values,l43.values,l5.values))
a_it6.name = 'aIT6'

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

# posterior ventral occipito-temporal
fus_1 = labels[label_keys['fusiform_1-lh']]
fus_2 = labels[label_keys['fusiform_2-lh']]
lat_occ_10 = labels[label_keys['lateraloccipital_10-lh']]
lat_occ_11 = labels[label_keys['lateraloccipital_11-lh']]
PV = MV.copy()
PV.vertices = np.concatenate((fus_1.vertices, fus_2.vertices, lat_occ_10.vertices,lat_occ_11.vertices))
PV.values = np.concatenate((fus_1.values, fus_2.values,lat_occ_10.values,lat_occ_11.values))
PV.name = 'PV'

# anterior ventral temporal
it_3 = labels[label_keys['inferiortemporal_3-lh']]

it_4 = labels[label_keys['inferiortemporal_4-lh']]
AV = it_3.copy()
AV.vertices = np.concatenate((it_3.vertices, it_4.vertices, a_it6.vertices))#anterior_it_6.vertices
AV.values = np.concatenate((it_3.values, it_4.values, a_it6.values))#, anterior_it_6.values
AV.name = 'AV'

# superior parietal
sup_par_10 = labels[label_keys['superiorparietal_10-lh']]
sup_par_11 = labels[label_keys['superiorparietal_11-lh']]
sup_par_12 = labels[label_keys['superiorparietal_12-lh']]
sup_par_13 = labels[label_keys['superiorparietal_13-lh']]
SP = AV.copy()
SP.vertices = np.concatenate((sup_par_10.vertices, sup_par_11.vertices, sup_par_12.vertices, sup_par_13.vertices))
SP.values = np.concatenate((sup_par_10.values, sup_par_11.values, sup_par_12.values, sup_par_13.values))
SP.name = 'SPT'

# inferior parietal (IPT)
inf_par_1 = labels[label_keys['inferiorparietal_1-lh']]
inf_par_2 = labels[label_keys['inferiorparietal_2-lh']]
inf_par_3 = labels[label_keys['inferiorparietal_3-lh']]
inf_par_4 = labels[label_keys['inferiorparietal_4-lh']]
inf_par_7 = labels[label_keys['inferiorparietal_7-lh']]
inf_par_9 = labels[label_keys['inferiorparietal_9-lh']]

posterior_inf_par_3, anterior_inf_par_3 = inf_par_3.split(('posterior', 'anterior'))

posterior_inf_par_1, anterior_inf_par_1 = inf_par_1.split(('posterior', 'anterior'))
IP = SP.copy()
IP.vertices = np.concatenate((inf_par_1.vertices, inf_par_2.vertices, inf_par_3.vertices, inf_par_4.vertices, inf_par_7.vertices, inf_par_9.vertices))
IP.values = np.concatenate((inf_par_1.values, inf_par_2.values, inf_par_3.values, inf_par_4.values, inf_par_7.values, inf_par_9.values))
IP.name = 'IPT'

# define list ROIs for visualization
rois = [SPM, LO, IPM, PV, SP, MV, IP, AV]


# 2. Extract time courses for each condition

# define data folders
data_folder = op.join('..', '..', 'source_activity', 'natural_reading')
method = 'MNE'
source_folder = op.join(data_folder, method)

src_fname = op.join(mne.datasets.fetch_fsaverage(), 'bem', 'fsaverage-ico-5-src.fif')
src = mne.read_source_spaces(src_fname, verbose=True)
subjects_dir = mne.datasets.sample.data_path() + '/subjects'
stc_file = op.join(data_folder, method, 'average')
average_stc = mne.read_source_estimate(stc_file, 'fsaverage')

# morphing steps:
# 1. fsaverage -> fsaverage_sym
# 2. left to right and vice versa
# 3. fsaverage_sym -> fsaverage (for labels)
av_stc_xhemi = mne.compute_source_morph(average_stc, 'fsaverage', 'fsaverage_sym', smooth=5,
                               warn=False,
                               subjects_dir=subjects_dir).apply(average_stc)
av_stc_xhemi = mne.compute_source_morph(av_stc_xhemi, 'fsaverage_sym', 'fsaverage_sym',
                                 spacing=average_stc.vertices, warn=False,
                                 subjects_dir=subjects_dir, xhemi=True,
                                 verbose='error').apply(av_stc_xhemi)
av_stc_xhemi = mne.compute_source_morph(av_stc_xhemi,'fsaverage_sym', 'fsaverage', smooth=5,
                               warn=False,
                               subjects_dir=subjects_dir).apply(av_stc_xhemi)


# morph and extract for all subjects
tmin=-0.0
tmax=0.350

subjects = ['842608', '587631', '217720', '059694', '394107', '356349', '044846', '050120', '269257',
            '103183', '862169', '284297', '643438', '048298', '414822', '638838', '390744', '930517',
            '093925', '213103', '331536', '205515', '135230', '320268', '319897', '321319', '303786',
            '822816', '163753', '667207', '424174', '612370', '528213', '009833', '927179', '515155',
            '366394', '133288']
n_subjects = len(subjects)

mode='mean'
roi = {}
xroi_dict = {}
stc_data = []
stc_xhemi_data = []
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
    stc_data.append(roi_data)
    stc_xhemi_data.append(xroi_data)

# create array and calculate STD and SEM
stc_data=np.array(stc_data)
stc_xhemi_data=np.array(stc_xhemi_data)
stc_std = []
stc_sem = []
xhemi_std = []
xhemi_sem = []
for i in range(len(rois)):
    print(i)
    stc_std.append(np.std(stc_data[:,i,:],axis=0))
    stc_sem.append(sp.sem(stc_data[:,i,:],axis=0, ddof=1))
    
    xhemi_std.append(np.std(stc_xhemi_data[:,i,:],axis=0))
    xhemi_sem.append(sp.sem(stc_xhemi_data[:,i,:],axis=0, ddof=1))

# plot
figure_roi = plt.figure(figsize=(6.5*cm,11*cm), dpi=142.12)
widths = [1,1]
heights = [1,1,1,1]
gs_rois = figure_roi.add_gridspec(ncols=2, nrows=4, width_ratios=widths,
                          height_ratios=heights, left=0.15, right=0.970, bottom=0.1)
gs_rois.update(wspace=0.5, hspace=0.3)

# set different exponents for each ROI
exponents = {LO.name:10**13, SPM.name:10**14, IPM.name:10**14, IP.name:10**13, SP.name:10**13, AV.name:10**13, MV.name:10**13, PV.name:10**13}

for i, label in enumerate(rois):
    axs1 = plt.subplot(gs_rois[i])
    x_roi_timecourse_fig(average_stc, av_stc_xhemi, src,  label, mode='mean', ax=axs1, exponent=exponents[label.name])#
    
    axs1.yaxis.offsetText.set_fontsize(6)


    axs1.set_ylabel('{} MSA\n(x e-13 a.u.)'.format(label.name, exponents[label.name]), size=6, labelpad=0.6)
    
    if i > 5:
        axs1.set_xlabel('Time (ms)', fontsize=8, labelpad=0.5) #
        
    if i == 7:
        plt.legend(fontsize=6, markerscale=0.1, prop={'size': 6}, frameon=False)

# add horizontal lines to the bottom to mark time windows
axs = figure_roi.get_axes()
axs[0].set_ylim(ymin=0,ymax=12.5)
axs[1].set_ylim(ymin=0,ymax=10.5)
axs[2].set_ylim(ymin=0,ymax=12)
axs[3].set_ylim(ymin=0,ymax=7)
axs[4].set_ylim(ymin=0,ymax=5)
axs[5].set_ylim(ymin=0,ymax=5)
axs[6].set_ylim(ymin=0,ymax=5)
axs[7].set_ylim(ymin=0,ymax=2.5)

axs[6].plot(np.arange(140,160,1),np.full(20, 0.25),  'k', linewidth=lw)
axs[6].plot(np.arange(175,225,1),np.full(50, 0.25),  color = (230/255.0, 159/255.0, 0), linewidth=lw)
axs[6].plot(np.arange(245,295,1),np.full(50, 0.25),  color = (0, 158/255.0, 115/255.0), linewidth=lw)
axs[7].plot(np.arange(140,160,1),np.full(20, 0.25/2),  'k-', linewidth=lw)
axs[7].plot(np.arange(175,225,1),np.full(50, 0.25/2),  color = (230/255.0, 159/255.0, 0), linewidth=lw)
axs[7].plot(np.arange(245,295,1),np.full(50, 0.25/2),  color = (0, 158/255.0, 115/255.0), linewidth=lw)

plt.show()
matplotlib.rcParams.update(new_rc_params)
figure_roi.savefig('nr_rois_av_with_sem_1014_ddof1.svg', figsize=(6.5*cm,11*cm), dpi=600)





