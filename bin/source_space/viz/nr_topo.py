# Create topographical visualizations for Natural Reading results
import os.path as op
import pickle
import numpy as np
import mne
import matplotlib as mpl
import matplotlib.pyplot as plt
mne.viz.set_3d_backend('pyvista')

# figure version setting
date='1014'

# some matplotlib and plotting parameters
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

cm = 1/2.54
plt.rcParams['figure.dpi'] = 142
plt.rcParams["figure.figsize"] = (18.0*cm,12.0*cm) 
font = {'family' : 'Arial',
        'weight' : 'bold',
        'size'   : 8}
plt.style.use('default')
#views for figures
views=['lateral','caudal','ventral']
tpoint_lims=[0.0, 2., 8.11]
bgc='1.0'
cmap='RdBu_r'
c_u=6 # crop 1/6th of upper part of image
c_l=7 # crop 1/7th of upper part of image
lw=1.5

# custom ventral view
ventral_view = dict(azimuth=0,elevation=205, roll=90)

# define functions for plotting and extracting data
def extract_min_and_max_t(T_obs, clusters, ind, n_vertices, n_times):
    v_inds = clusters[ind][1]
    t_inds = clusters[ind][0]
    data = np.zeros((n_vertices, n_times))
    data[v_inds, t_inds] = T_obs[t_inds, v_inds]
    
    return np.min(data), np.max(data)


def get_label_from_vertex(vertex, labels, hemi='-lh'):

    for label in [l for l in labels if (hemi in l.name)]:
        if vertex in label.get_vertices_used():
            return(label.name)
            break


def show_cluster_timecourse(clusters, T_obs, stc, ind, n_vertices, n_times, hemi='left_hemi'):
    stc = stc.copy()
    stc.data = np.zeros(stc.data.shape)
    v_inds = clusters[ind][1]
    t_inds = clusters[ind][0]
    data = np.zeros((n_vertices, n_times))
    data[v_inds, t_inds] = T_obs[t_inds, v_inds]
    
    h='lh'
    if hemi == 'left_hemi':
        stc.data[:n_vertices, :n_times] = data
    else:
        h='rh'
        stc.data[n_vertices:, :n_times] = data
        
    max_t_vertex, max_t_latency = stc.get_peak(mode='abs')
    max_t_vind, max_t_lind = stc.get_peak(vert_as_index=True, time_as_index=True)
    print('peak vertex: {}, peak latency: {}'.format(max_t_vertex, max_t_latency))
    print(mne.vertex_to_mni(max_t_vertex, 0, stc.subject))
    print('max_t: {:.3f}'.format(stc.data[max_t_vind][max_t_lind]))
    print(get_label_from_vertex(max_t_vertex, labels))
    
    views =['lateral', 'caudal', 'ventral', 'medial']
    tview_kwargs = dict(hemi=h,views=views, #surface='white', #clim=dict(kind='value', lims=lims),
                    time_unit='s', smoothing_steps=5, colorbar = True, size = 900, time_viewer = True,
                    colormap='auto', view_layout='horizontal', cortex='bone')
    brain = stc.plot(**tview_kwargs)
    return brain, stc


def brain_snapshot(stc, views, t, h='lh', cb=False, lims=[0.0, 2.3, 10.0], cmap=cmap, roi=None):
    stc = stc.copy()
    stc.crop(tmin=t, tmax=t)
    views =views
    tview_kwargs = dict(hemi=h, initial_time=t, clim=dict(kind='value', pos_lims=lims), colormap=cmap, 
                    time_unit='s', smoothing_steps=5, colorbar = cb, size = 900, time_viewer = False,
                    view_layout='horizontal', cortex='bone', background=(1.,1.,1.,0.))
    brain = stc.plot(**tview_kwargs)
    if roi:
        colors=['black']*10
        #colors=['gold', 'darkorange', 'black', 'olive', 'darkgreen', 'saddlebrown', 'darkslateblue', 'purple']
        for i,r in enumerate(roi):
            brain.add_label(r, color=colors[i], borders=True)
    brain.show_view(view=views)
    return brain


def visualize_cluster_summary(clu, stc_file, method, time_start,time_end, hemi='left_hemi'):
    T_obs, clusters, cluster_p_values, H0 = clu
    stc = mne.read_source_estimate(stc_file, 'fsaverage')
    stc_all_cluster_vis = mne.stats.summarize_clusters_stc(clu, tstep=stc.tstep,
                                             vertices=[np.arange(T_obs.shape[1])],
                                             subject='fsaverage')

    stc.data = np.zeros((stc.data.shape[0], stc_all_cluster_vis.data.shape[1]))
    stc.data[:T_obs.shape[1],:] = stc_all_cluster_vis.data
    
    return stc

# 1. Create custom labels
subjects_dir = mne.datasets.fetch_fsaverage()
src_sym = op.join('..', '..','fsaverage_sym','bem', 'fsaverage_sym_ico5-src.fif')
src_sym = mne.read_source_spaces(src_sym)
src_fsav = src_fname = op.join(mne.datasets.fetch_fsaverage(), 'bem', 'fsaverage-ico-5-src.fif')
src_fsav = mne.read_source_spaces(src_fsav)
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir,
                                        verbose=True)

mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=subjects_dir,
                                          verbose=True)

hcp_labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'lh', subjects_dir=subjects_dir)

l_fef = hcp_labels[62]
l_55b = hcp_labels[29]
hcp_label_keys = {hcp_label.name: i for i,hcp_label in enumerate(hcp_labels)}

labels = mne.read_labels_from_annot('fsaverage', 'aparc_sub', subjects_dir=subjects_dir)
label_keys = {label.name: i for i,label in enumerate(labels)}


SPM = l_fef.copy()
SPM.vertices = np.concatenate((l_fef.vertices,l_55b.vertices))
SPM.values = np.concatenate((l_fef.values,l_55b.values))
SPM.name = 'SPM'


IPM = SPM.copy()
L_6r = hcp_labels[hcp_label_keys['L_6r_ROI-lh']]
L_6v = hcp_labels[hcp_label_keys['L_6v_ROI-lh']]
L_IFJa = hcp_labels[hcp_label_keys['L_IFJa_ROI-lh']]
L_IFJp = hcp_labels[hcp_label_keys['L_IFJp_ROI-lh']]
dorsal_L_6v, ventral_L_6v =  L_6v.split(('dorsal', 'ventral'))
IPM.vertices = np.concatenate((L_6r.vertices, L_6v.vertices, L_IFJa.vertices, L_IFJp.vertices))
IPM.values = np.concatenate((L_6r.values, L_6v.values, L_IFJa.values,L_IFJp.values))
IPM.name = 'IPM'


it_6 = labels[label_keys['inferiortemporal_6-lh']]
l1,l2,l3,l4,l5 = it_6.split(5)
l31, l32 = l3.split(2)
l41, l42, l43 = l4.split(3)
p_it6 = IPM.copy()
p_it6.vertices = np.concatenate((l1.vertices,l2.vertices,l31.vertices,l41.vertices))
p_it6.values = np.concatenate((l1.values,l2.values,l31.values,l41.values))
p_it6.name = 'pIT6'

a_it6 = IPM.copy()
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

fus_1 = labels[label_keys['fusiform_1-lh']]
fus_2 = labels[label_keys['fusiform_2-lh']]
lat_occ_10 = labels[label_keys['lateraloccipital_10-lh']]
lat_occ_11 = labels[label_keys['lateraloccipital_11-lh']]
PV = MV.copy()
PV.vertices = np.concatenate((fus_1.vertices, fus_2.vertices, lat_occ_10.vertices,lat_occ_11.vertices))
PV.values = np.concatenate((fus_1.values, fus_2.values,lat_occ_10.values,lat_occ_11.values))
PV.name = 'PV'

lat_occ_2 = labels[label_keys['lateraloccipital_2-lh']]
lat_occ_3 = labels[label_keys['lateraloccipital_3-lh']]
LO = lat_occ_2.copy()
LO.vertices = np.concatenate((lat_occ_2.vertices, lat_occ_3.vertices))
LO.values = np.concatenate((lat_occ_2.values, lat_occ_3.values))
LO.name = 'LO'


it_3 = labels[label_keys['inferiortemporal_3-lh']]

it_4 = labels[label_keys['inferiortemporal_4-lh']]
AV = it_3.copy()
AV.vertices = np.concatenate((it_3.vertices, it_4.vertices, a_it6.vertices))#anterior_it_6.vertices
AV.values = np.concatenate((it_3.values, it_4.values, a_it6.values))#, anterior_it_6.values
AV.name = 'AV'


lat_occ_7 = labels[label_keys['lateraloccipital_7-lh']]
posterior_lat_occ_7, anterior_lat_occ_7 = lat_occ_7.split(('posterior', 'anterior'))
lat_occ_8 = labels[label_keys['lateraloccipital_8-lh']]

mid_temp_1 = labels[label_keys['middletemporal_1-lh']]
mid_temp_2 = labels[label_keys['middletemporal_2-lh']]

LP = AV.copy()
LP.vertices = np.concatenate((lat_occ_7.vertices, lat_occ_8.vertices, dorsal_lat_occ_9.vertices, mid_temp_1.vertices, mid_temp_2.vertices, dorsal_it_8.vertices))
LP.values = np.concatenate((lat_occ_7.values, lat_occ_8.values, dorsal_lat_occ_9.values, mid_temp_1.values, mid_temp_2.values, dorsal_it_8.values))
LP.name = 'LP'



# inferior parietal
inf_par_1 = labels[label_keys['inferiorparietal_1-lh']]
inf_par_2 = labels[label_keys['inferiorparietal_2-lh']]
inf_par_3 = labels[label_keys['inferiorparietal_3-lh']]
inf_par_4 = labels[label_keys['inferiorparietal_4-lh']]
inf_par_7 = labels[label_keys['inferiorparietal_7-lh']]
inf_par_9 = labels[label_keys['inferiorparietal_9-lh']]

posterior_inf_par_3, anterior_inf_par_3 = inf_par_3.split(('posterior', 'anterior'))

posterior_inf_par_1, anterior_inf_par_1 = inf_par_1.split(('posterior', 'anterior'))
IP = LP.copy()
IP.vertices = np.concatenate((inf_par_1.vertices, inf_par_2.vertices, inf_par_3.vertices, inf_par_4.vertices, inf_par_7.vertices, inf_par_9.vertices))
IP.values = np.concatenate((inf_par_1.values, inf_par_2.values, inf_par_3.values, inf_par_4.values, inf_par_7.values, inf_par_9.values))
IP.name = 'IPT'


sup_par_10 = labels[label_keys['superiorparietal_10-lh']]
sup_par_11 = labels[label_keys['superiorparietal_11-lh']]
sup_par_12 = labels[label_keys['superiorparietal_12-lh']]
sup_par_13 = labels[label_keys['superiorparietal_13-lh']]
SP = IP.copy()
SP.vertices = np.concatenate((sup_par_10.vertices, sup_par_11.vertices, sup_par_12.vertices, sup_par_13.vertices))
SP.values = np.concatenate((sup_par_10.values, sup_par_11.values, sup_par_12.values, sup_par_13.values))
SP.name = 'SPT'

# define list of ROIs
rois = [SPM, LO, IPM, PV, SP, MV, IP, AV]

# paths and data
data_folder = op.join('..', '..', 'source_activity', 'natural_reading')
result_folder = op.join('..', '..', 'results', 'natural_reading')

method = 'MNE'
hemi = 'left_hemi'
p_threshold = 0.05
p_threshold = str(p_threshold)
n_permutations = 1000
n_permutations= str(int(n_permutations))


# time window 140-160ms

tmin = 0.140
tmax = 0.160
time_start = str(int(tmin*1000))
time_end = str(int(tmax*1000))
stc_file = op.join(data_folder, method, 'average')
stc = mne.read_source_estimate(stc_file, 'fsaverage_sym').crop(tmin,tmax)
stc.data = np.zeros(stc.data.shape)

# Load pickle file
fname = 'sym_src_normalized_contrast_'+time_start + '-' + time_end + 'ms' + '_nperm-1024_all_clu.pkl'
pickle_file = op.join(result_folder, hemi, method, 'cluster_timecourse', fname)

T_obs, clusters, cluster_p_values, H0 = clu = pickle.load( open( pickle_file, "rb" ) )
n_times, n_vertices = T_obs.shape

# Get cluster indeces and print p vals
alpha = 0.15
good_cluster_inds = np.where(cluster_p_values <= alpha)[0]
print(good_cluster_inds)
if len(good_cluster_inds) > 0:
    print('Good cluster indeces: {}'.format(good_cluster_inds))
    for i in good_cluster_inds:
        print('Cluster {}: p<= {}'.format(i, cluster_p_values[i]))
elif len(good_cluster_inds)==0:
    print('No clusters significant at p <= {}'.format(alpha))

clu1_brain, clu1_stc_sym = show_cluster_timecourse(clusters, T_obs, stc, 1, n_vertices, n_times, hemi=hemi)

# Compute a morph-matrix mapping the right to the left hemisphere,
# and vice-versa.
morph = mne.compute_source_morph(src_sym, 'fsaverage_sym', 'fsaverage',
                                 spacing=clu1_stc_sym.vertices, warn=False,
                                 subjects_dir=None, xhemi=False, src_to=src_fsav,
                                 verbose='error') 
clu1_stc_sym_to_av = morph.apply(clu1_stc_sym)

max_t_vertex, max_t_latency = clu1_stc_sym_to_av.get_peak(mode='abs')
max_t_vind, max_t_lind = clu1_stc_sym_to_av.get_peak(vert_as_index=True, time_as_index=True)
print('peak vertex: {}, peak latency: {}'.format(max_t_vertex, max_t_latency))
print(mne.vertex_to_mni(max_t_vertex, 0, clu1_stc_sym_to_av.subject))
print('max_t: {:.3f}'.format(clu1_stc_sym_to_av.data[max_t_vind][max_t_lind]))
print(get_label_from_vertex(max_t_vertex, labels))

tview_kwargs = dict(hemi='lh',views=views, #surface='white', #clim=dict(kind='value', lims=lims),
                time_unit='s', smoothing_steps=5, colorbar = True, size = 900, time_viewer = True,
                colormap='auto', view_layout='horizontal', cortex='bone')
brain = clu1_stc_sym_to_av.plot(**tview_kwargs)


clu4_brain, clu4_stc = show_cluster_timecourse(clusters, T_obs, stc, 4, n_vertices, n_times, hemi=hemi)

# Compute a morph-matrix mapping the right to the left hemisphere,
# and vice-versa.
morph = mne.compute_source_morph(clu4_stc, 'fsaverage_sym', 'fsaverage',
                                 spacing=clu4_stc.vertices, warn=False,
                                 subjects_dir=None, xhemi=False,
                                 verbose='error') 
clu4_stc_sym_to_av = morph.apply(clu4_stc)

max_t_vertex, max_t_latency = clu4_stc_sym_to_av.get_peak(mode='abs')
max_t_vind, max_t_lind = clu4_stc_sym_to_av.get_peak(vert_as_index=True, time_as_index=True)
print('peak vertex: {}, peak latency: {}'.format(max_t_vertex, max_t_latency))
print(mne.vertex_to_mni(max_t_vertex, 0, clu4_stc_sym_to_av.subject))
print('max_t: {:.3f}'.format(clu4_stc_sym_to_av.data[max_t_vind][max_t_lind]))
print(get_label_from_vertex(max_t_vertex, labels))

tview_kwargs = dict(hemi='lh',views=views, #clim=dict(kind='value', lims=lims),
                time_unit='s', smoothing_steps=5, colorbar = True, size = 900, time_viewer = True,
                colormap='auto', view_layout='horizontal', cortex='bone')

pooled_stc = clu1_stc_sym_to_av.copy()
pooled_stc.data = clu1_stc_sym_to_av.data + clu4_stc_sym_to_av.data

brain = pooled_stc.plot(view_layout='horizontal', views=['lateral', 'ventral', 'caudal'])


kwargs = dict(azimuth=0,elevation=200, roll=90)
brain.show_view(view=kwargs)


mean_140_160 = pooled_stc.copy()

mean_140_160.data = mean_140_160.data.mean(axis=1).reshape(20484, 1)
max_t_vind, max_t_lind = mean_140_160.get_peak(vert_as_index=True, time_as_index=True)
print(mean_140_160.data[max_t_vind, max_t_lind])


max_t_vind, max_t_lind = mean_140_160.get_peak(vert_as_index=True, time_as_index=True)
mean_140_160.data[max_t_vind][max_t_lind]


mean_images_140_160 = []
for v in views:
    l = rois 
    if v == 'ventral':
        brain = brain_snapshot(mean_140_160, views=ventral_view, t=0.140, lims=tpoint_lims, roi=l)
    else:
        brain = brain_snapshot(mean_140_160, views=v, t=0.140, lims=tpoint_lims, roi=l)
    
    img = brain.screenshot(time_viewer=False, mode='rgba')
    lx, ly,a = img.shape
    # Cropping 
    img = img[lx // c_u: - lx// c_l, :, :]
    
    mean_images_140_160.append(img)


# time window 175-225ms

tmin = 0.175
tmax = 0.225
time_start = str(int(tmin*1000))
time_end = str(int(tmax*1000))
stc_file = op.join(data_folder, method, 'average')
stc = mne.read_source_estimate(stc_file, 'fsaverage_sym').crop(tmin,tmax)
stc.data = np.zeros(stc.data.shape)

# Load pickle file
fname = 'sym_src_normalized_contrast_'+time_start + '-' + time_end + 'ms' + '_nperm-1024_all_clu.pkl'
pickle_file = op.join(result_folder, hemi, method, 'cluster_timecourse', fname)

T_obs, clusters, cluster_p_values, H0 = clu = pickle.load( open( pickle_file, "rb" ) )
n_times, n_vertices = T_obs.shape

# Get cluster indeces and print p vals
alpha = 0.15
good_cluster_inds = np.where(cluster_p_values <= alpha)[0]
print(good_cluster_inds)
if len(good_cluster_inds) > 0:
    print('Good cluster indeces: {}'.format(good_cluster_inds))
    for i in good_cluster_inds:
        print('Cluster {}: p<= {}'.format(i, cluster_p_values[i]))
elif len(good_cluster_inds)==0:
    print('No clusters significant at p <= {}'.format(alpha))

brain, clu335_stc = show_cluster_timecourse(clusters, T_obs, stc, 5, n_vertices, n_times, hemi=hemi)


# Compute a morph-matrix mapping the right to the left hemisphere,
# and vice-versa.
morph = mne.compute_source_morph(clu335_stc, 'fsaverage_sym', 'fsaverage',
                                 spacing=clu335_stc.vertices, warn=False,
                                 subjects_dir=None, xhemi=False,
                                 verbose='error') 
clu335_stc_sym_to_av = morph.apply(clu335_stc)

max_t_vertex, max_t_latency = clu335_stc_sym_to_av.get_peak(mode='abs')
max_t_vind, max_t_lind = clu335_stc_sym_to_av.get_peak(vert_as_index=True, time_as_index=True)
print('peak vertex: {}, peak latency: {}'.format(max_t_vertex, max_t_latency))
print(mne.vertex_to_mni(max_t_vertex, 0, clu335_stc_sym_to_av.subject))
print('max_t: {:.3f}'.format(clu335_stc_sym_to_av.data[max_t_vind][max_t_lind]))
print(get_label_from_vertex(max_t_vertex, labels))


tview_kwargs = dict(hemi='lh',views='lateral', #clim=dict(kind='value', lims=lims),
                time_unit='s', smoothing_steps=5, colorbar = True, size = 900, time_viewer = True,
                colormap='auto', view_layout='horizontal', cortex='bone')
brain = clu335_stc_sym_to_av.plot(**tview_kwargs)

mean_175_225 = clu335_stc_sym_to_av.copy()
mean_175_225.data = mean_175_225.data.mean(axis=1).reshape(20484, 1)

max_t_vind, max_t_lind = mean_175_225.get_peak(vert_as_index=True, time_as_index=True)
print(mean_175_225.data[max_t_vind][max_t_lind])


mean_images_175_225 = []
for v in views:
    l = rois
    if v == 'ventral':
        brain = brain_snapshot(mean_175_225, views=ventral_view, t=0.175, lims=tpoint_lims, roi=l)
    else:
        brain = brain_snapshot(mean_175_225, views=v, t=0.175, lims=tpoint_lims, roi=l)
    
    img = brain.screenshot(time_viewer=False, mode='rgba') 
    lx, ly,a = img.shape
    # Cropping 
    img = img[lx // c_u: - lx// c_l, :, :]
    
    mean_images_175_225.append(img)


# time window 245-295ms
tmin = 0.245
tmax = 0.295
time_start = str(int(tmin*1000))
time_end = str(int(tmax*1000))
stc_file = op.join(data_folder, method, 'average')
stc = mne.read_source_estimate(stc_file, 'fsaverage_sym').crop(tmin,tmax)
stc.data = np.zeros(stc.data.shape)

# Load pickle file
fname = 'normalized_contrast_'+time_start + '-' + time_end + 'ms' + '_nperm-' + n_permutations + '_all_clu.pkl'
pickle_file = op.join(result_folder, hemi, method, 'cluster_timecourse', fname)

T_obs, clusters, cluster_p_values, H0 = clu = pickle.load( open( pickle_file, "rb" ) )
n_times, n_vertices = T_obs.shape

# Get cluster indeces and print p vals
alpha = 0.05
good_cluster_inds = np.where(cluster_p_values <= alpha)[0]
print(good_cluster_inds)
if len(good_cluster_inds) > 0:
    print('Good cluster indeces: {}'.format(good_cluster_inds))
    for i in good_cluster_inds:
        print('Cluster {}: p<= {}'.format(i, cluster_p_values[i]))
elif len(good_cluster_inds)==0:
    print('No clusters significant at p <= {}'.format(alpha))

brain, clu369_stc = show_cluster_timecourse(clusters, T_obs, stc, 369, n_vertices, n_times, hemi=hemi)


# Compute a morph-matrix mapping the right to the left hemisphere,
# and vice-versa.
morph = mne.compute_source_morph(clu369_stc, 'fsaverage_sym', 'fsaverage',
                                 spacing=clu369_stc.vertices, warn=False,
                                 subjects_dir=None, xhemi=False,
                                 verbose='error') 
clu369_stc_sym_to_av = morph.apply(clu369_stc)

max_t_vertex, max_t_latency = clu369_stc_sym_to_av.get_peak(mode='abs')
max_t_vind, max_t_lind = clu369_stc_sym_to_av.get_peak(vert_as_index=True, time_as_index=True)
print('peak vertex: {}, peak latency: {}'.format(max_t_vertex, max_t_latency))
print(mne.vertex_to_mni(max_t_vertex, 0, clu369_stc_sym_to_av.subject))
print('max_t: {:.3f}'.format(clu369_stc_sym_to_av.data[max_t_vind][max_t_lind]))
print(get_label_from_vertex(max_t_vertex, labels))

tview_kwargs = dict(hemi='lh',views='lateral', #clim=dict(kind='value', lims=lims),
                time_unit='s', smoothing_steps=5, colorbar = True, size = 900, time_viewer = True,
                colormap='auto', view_layout='horizontal', cortex='bone')

mean_245_295 = clu369_stc_sym_to_av.copy()
mean_245_295.data = mean_245_295.data.mean(axis=1).reshape(20484, 1)

max_t_vind, max_t_lind = mean_245_295.get_peak(vert_as_index=True, time_as_index=True)
print(mean_245_295.data[max_t_vind][max_t_lind])


mean_images_245_295 = []
for v in views:
    l = rois
    if v == 'ventral':
        brain = brain_snapshot(mean_245_295, views=ventral_view, t=0.245, lims=tpoint_lims, roi=l)
    else:
        brain = brain_snapshot(mean_245_295, views=v, t=0.245, lims=tpoint_lims, roi=l)
    
    img = brain.screenshot(time_viewer=False, mode='rgba') 
    lx, ly,a = img.shape
    # Cropping 
    img = img[lx // c_u: - lx// c_l, :, :]
    mean_images_245_295.append(img)


# Summary figure with subplots
# Mean across time * views

images = [mean_images_140_160, mean_images_175_225, mean_images_245_295]

widths = [3, 3, 3]
heights = [3, 3, 3]

figure_mean = plt.figure(figsize=(17.5*cm,17.5*cm), dpi=142.12)
gs_mean = figure_mean.add_gridspec(ncols=3, nrows=3, width_ratios=widths,
                          height_ratios=heights, left=0.03, right=0.85, bottom=0.10001)

gs_mean.update(wspace=0.0, hspace=0.00) # set the spacing between axes. 
for col in range(4):
    for row, title in zip([0,1,2], ['140-160 ms', '175-225 ms', '245-290 ms']):
        if col < 3:
            axs1 = plt.subplot(gs_mean[row, col])
            im = axs1.imshow(images[row][col])
            axs1.set_axis_off()
            axs1.set_aspect('equal')
        if col==0:
            axs1.set_title(title, x=-0.05, y=0.30, rotation='vertical', fontsize=8)
    
figure_mean.suptitle('Mean cluster t-value across time window', fontsize=8, fontweight='bold')

colorbar='h'
cbar = None
if colorbar == 'v':
    widths = [1]
    heights = [1, 1, 1, 1, 1,1,1,1,1]
    gs_cb = figure_mean.add_gridspec(ncols=1, nrows=9, width_ratios=widths,
                                   height_ratios=heights, top=0.85,bottom=0.25,
                                   left=0.94, right=0.95)
    ax = plt.subplot(gs_cb[3:9, 0])
    cbar = mne.viz.plot_brain_colorbar(ax, clim=dict(kind='value', pos_lims=tpoint_lims), orientation='vertical',
                                label='t-value', bgcolor=bgc, colormap=cmap)
elif colorbar=='h':
    heights = [1]
    widths = [1, 1, 1, 1, 1,1,1,1,1]
    gs_cb = figure_mean.add_gridspec(ncols=9, nrows=1, width_ratios=widths,
                                   height_ratios=heights, top=0.10,bottom=0.09,
                                   left=0.0, right=1)
    ax = plt.subplot(gs_cb[0, 2:6])
    cbar = mne.viz.plot_brain_colorbar(ax, clim=dict(kind='value', pos_lims=tpoint_lims), orientation='horizontal',
                                label='t-value', bgcolor=bgc, colormap=cmap)
    
cbar.ax.tick_params(labelsize=6, pad=1) 
cbar.set_ticks([-8,-6,-4,-2,0,2,4,6,8], update_ticks=True)
cbar.set_label(label= 't-value', size=6,weight='bold')


mpl.rcParams.update(new_rc_params)
figure_mean.savefig('nr_average_'+date+'_max_t.svg', figsize=(17.5*cm,17,5*cm), dpi=600)




