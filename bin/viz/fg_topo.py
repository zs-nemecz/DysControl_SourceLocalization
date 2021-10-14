# Create topographical visualizations for Fixed Gazed results
import os.path as op
import pickle
import numpy as np
import mne
mne.viz.set_3d_backend('pyvista')
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})


# for saving figure versions
date='1014'

# figure and matplotlib parameters
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)
cm = 1/2.54
plt.rcParams['figure.dpi'] = 142
font = {'family' : 'Arial',
        'weight' : 'bold',
        'size'   : 8}
plt.style.use('default')
bgc='1.0'
cmap='RdBu_r'
c_u=6 # crop 1/6th of upper part of snapshot image
c_l=7 # crop 1/7th of upper part of snapshot image
lw=1.5

views=['lateral','caudal','ventral']

subjects_dir = op.join(mne.datasets.sample.data_path(),'subjects')
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir,
                                        verbose=True)

mne.datasets.fetch_aparc_sub_parcellation(subjects_dir=subjects_dir,
                                          verbose=True)

# functions for extracting data and viz
def show_cluster_p_vals(alpha, cluster_p_values):    
    good_cluster_inds = np.where(cluster_p_values <= alpha)[0]

    if len(good_cluster_inds) > 0:
        print('Good cluster indeces: {}'.format(good_cluster_inds))
        for i in good_cluster_inds:
            print('Cluster {}: p<= {}'.format(i, cluster_p_values[i]))
    else:
        print('No clusters significant at p <= {}'.format(alpha))


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
    
    views =['lateral']
    tview_kwargs = dict(hemi=h,views=views, #surface='white', #clim=dict(kind='value', lims=lims),
                    time_unit='s', smoothing_steps=5, colorbar = True, size = 900, time_viewer = True,
                    colormap='auto', view_layout='horizontal', cortex='bone')
    brain = stc.plot(**tview_kwargs)
    return brain, stc


def visualize_cluster_summary(clu, stc_file, method, time_start,time_end, hemi='left_hemi'):
    T_obs, clusters, cluster_p_values, H0 = clu
    stc = mne.read_source_estimate(stc_file, 'fsaverage')
    stc_all_cluster_vis = mne.stats.summarize_clusters_stc(clu, tstep=stc.tstep,
                                             vertices=[np.arange(T_obs.shape[1])],
                                             subject='fsaverage')

    stc.data = np.zeros((stc.data.shape[0], stc_all_cluster_vis.data.shape[1]))
    stc.data[:T_obs.shape[1],:] = stc_all_cluster_vis.data
    time_label = method +'\n'+ time_start + '-' + time_end + ' ms'
    
    h='lh'
    if hemi == 'right_hemi':
        h='rh'
    elif hemi=='both':
        h='both'
    
    title='Cluster summary'
    brain = stc.plot(hemi=h, views=['lateral', 'caudal', 'ventral', 'medial'], time_label=time_label, size=900,
                     smoothing_steps=5, time_viewer=True, view_layout='horizontal')
    return brain

def brain_snapshot(stc, views, t, h='lh', cb=False, lims=[0.0, 2.3, 10.0], cmap=cmap, roi=None):
    stc = stc.copy()
    stc.crop(tmin=t, tmax=t)
    views =views
    tview_kwargs = dict(hemi=h, initial_time=t, clim=dict(kind='value', pos_lims=lims), colormap=cmap, 
                    time_unit='s', smoothing_steps=5, colorbar = cb, size = 900, time_viewer = False,
                    view_layout='horizontal', cortex='bone', background=(1.,1.,1.,0.))
    brain = stc.plot(**tview_kwargs)
    if roi:
        for i,r in enumerate(roi):
            brain.add_label(r, color='black', borders=True)
    brain.show_view(view=views)
    return brain


# 1. Create custom labels from atlas aparc_sub and hcp
labels = mne.read_labels_from_annot('fsaverage', 'aparc_sub', subjects_dir=subjects_dir)
label_keys = {label.name: i for i,label in enumerate(labels)}


# lateral occipital
lat_occ_2 = labels[label_keys['lateraloccipital_2-lh']]
lat_occ_3 = labels[label_keys['lateraloccipital_3-lh']]
LO = lat_occ_2.copy()
LO.vertices = np.concatenate((lat_occ_2.vertices, lat_occ_3.vertices))
LO.values = np.concatenate((lat_occ_2.values, lat_occ_3.values))
LO.name = 'LO'

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

# anterior lateral occipital label
temporal_LP = LP.copy()
temporal_LP.vertices = np.concatenate((mid_temp_1.vertices, mid_temp_2.vertices, dorsal_it_8.vertices))
temporal_LP.values = np.concatenate((mid_temp_1.values, mid_temp_2.values, dorsal_it_8.values))
temporal_LP.name = 'PT'

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

# define ROI list
rois = [LO, IP, temporal_LP, occipital_LP, PV, MV]

# 2. Extract time courses for each condition

# define data folders
data_folder = op.join('..', '..', 'source_activity', 'fixed_gaze')
result_folder = op.join('..', '..','results', 'fixed_gaze', 'cross_hemi')
method = 'MNE'
hemi = 'left_hemi'
p_threshold = 0.05
p_threshold = str(p_threshold)
n_permutations = 1000
n_permutations= str(int(n_permutations))


# Normal Condition

# use these lims for Normal and Armenian condition figures (peak t-val of these condtions)
tpoint_lims=[0.0, 2.0, 8.32]

# cluster time window
tmin = 0.175
tmax = 0.335
time_label = str(int(tmin * 1000)) + '-' + str(int(tmax * 1000)) + 'ms'
condition = 'normal'
time_start = str(int(tmin*1000))
time_end = str(int(tmax*1000))

# create empty stc
stc_file = op.join(data_folder, condition, method, condition + '_average')
stc = mne.read_source_estimate(stc_file, 'fsaverage').crop(tmin,tmax)
stc.data = np.zeros(stc.data.shape)

# Load pickle file
fname = 'normalized_contrast_'+ condition + '_' + time_start + '-' + time_end + 'ms' + '_nperm-' + n_permutations + '_all_clu.pkl'
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
tc_brain, normal_stc = show_cluster_timecourse(clusters, T_obs, stc, 0, n_vertices, n_times, hemi=hemi)

# define custom ventral view
ventral_view = dict(azimuth=0,elevation=205, roll=90)

norm_av_lims=[0.,4, 12.,]
mean_175_220 = normal_stc.copy().crop(tmin=0.175,tmax=0.220)
mean_175_220.data = mean_175_220.data.mean(axis=1).reshape(20484, 1)
tview_kwargs = dict(hemi='lh', clim=dict(kind='value', pos_lims=norm_av_lims), colormap=cmap, 
                time_unit='s', smoothing_steps=5, size = 900, time_viewer = True,
                view_layout='horizontal', cortex='bone', background=(1.,1.,1.,0.))

# get peak of t-vals averaged across time
mean_175_220.get_peak(vert_as_index=True, time_as_index=True)
mean_175_220.data[7950,0]

# snapshots from time window 175-220
mean_images_175_220 = []
for v in views:
    l = rois
    if v == 'ventral':
        brain = brain_snapshot(mean_175_220, views=ventral_view, t=0.175, lims=tpoint_lims, roi=l)
    else:
        brain = brain_snapshot(mean_175_220, views=v, t=0.175, lims=tpoint_lims, roi=l)
    
    img = brain.screenshot(time_viewer=False, mode='rgba') 
    lx, ly,a = img.shape
    # Cropping 
    img = img[lx // c_u: - lx// c_l, :, :]
    
    mean_images_175_220.append(img)

mean_220_270 = normal_stc.copy().crop(tmin=0.220,tmax=0.290)
mean_220_270.data = mean_220_270.data.mean(axis=1).reshape(20484, 1)
vind, tind = mean_220_270.get_peak(vert_as_index=True, time_as_index=True)
print(mean_220_270.data[vind, tind])

# snapshots from time window 220-270
mean_images_220_270 = []
for v in views:
    l = rois
    if v == 'ventral':
        brain = brain_snapshot(mean_220_270, views=ventral_view, t=0.220, lims=tpoint_lims, roi=l)
    else:
        brain = brain_snapshot(mean_220_270, views=v, t=0.220, lims=tpoint_lims, roi=l)
    
    img = brain.screenshot(time_viewer=False, mode='rgba') 
    lx, ly,a = img.shape
    # Cropping 
    img = img[lx // c_u: - lx// c_l, :, :]
    
    mean_images_220_270.append(img)

# updating parameters again to solve svg issue
mpl.rcParams.update(new_rc_params)


images = [mean_images_175_220, mean_images_220_270]

widths = [3, 3, 3]
heights = [3, 3]

figure_av = plt.figure(figsize=(17.5*cm,17.5*cm*2/3), dpi=142.12)
gs_tp = figure_av.add_gridspec(ncols=3, nrows=2, width_ratios=widths,
                          height_ratios=heights, left=0.03, right=0.90, bottom=0.2)

gs_tp.update(wspace=0.0, hspace=0.00) # set the spacing between axes. 
for col in range(4):
    for row, title in zip([0,1,2], ['175-220 ms', '220-290 ms']):
        if col < 3:
            axs1 = plt.subplot(gs_tp[row, col])
            im = axs1.imshow(images[row][col])
            axs1.set_axis_off()
            axs1.set_aspect('equal')
        if col==0:
            axs1.set_title(title, x=-0.05, y=0.30, rotation='vertical', fontsize=8)
    
figure_av.suptitle('Normal Condition\nMean cluster t-value across time window', fontsize=8, fontweight='bold')

colorbar='v'
if colorbar == 'v':
    widths = [1]
    heights = [1, 1, 1, 1, 1,1,1,1,1]
    gs_cb = figure_av.add_gridspec(ncols=1, nrows=9, width_ratios=widths,
                                   height_ratios=heights, top=0.85,bottom=0.25,
                                   left=0.94, right=0.95)
    ax = plt.subplot(gs_cb[3:9, 0])
    cbar = mne.viz.plot_brain_colorbar(ax, clim=dict(kind='value', pos_lims=tpoint_lims), orientation='vertical',
                                label='t-value', bgcolor=bgc, colormap=cmap)
elif colorbar=='h':
    heights = [1]
    widths = [1, 1, 1, 1, 1,1,1,1,1]
    gs_cb = figure_av.add_gridspec(ncols=9, nrows=1, width_ratios=widths,
                                   height_ratios=heights, top=0.150,bottom=0.13,
                                   left=0.0, right=1)
    ax = plt.subplot(gs_cb[0, 2:6])
    cbar = mne.viz.plot_brain_colorbar(ax, clim=dict(kind='value', pos_lims=tpoint_lims), orientation='horizontal',
                                label='t-value', bgcolor=bgc, colormap=cmap)
    
cbar.ax.tick_params(labelsize=6, pad=1) 
cbar.set_ticks([-8,-6,-4,-2,0,2,4,6,8], update_ticks=True)
cbar.set_label(label= 't-value', size=6,weight='bold')

mpl.rcParams.update(new_rc_params)
figure_av.savefig('fg_normal_average_'+date + '_max-t.svg', figsize=(17.5*cm,17.5*cm*2/3), dpi=600)


# Armenian condition

tmin = 0.175
tmax = 0.335
condition = 'armenian'
time_start = str(int(tmin*1000))
time_end = str(int(tmax*1000))
stc_file = op.join(data_folder, condition, method, condition + '_average')
stc = mne.read_source_estimate(stc_file, 'fsaverage').crop(tmin,tmax)
stc.data = np.zeros(stc.data.shape)

# Load pickle file
fname = 'normalized_contrast_'+ condition + '_' + time_start + '-' + time_end + 'ms' + '_nperm-' + n_permutations + '_all_clu.pkl'
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

clu1, armenian_stc = show_cluster_timecourse(clusters, T_obs, stc, 1, n_vertices, n_times, hemi=hemi)
clu2 = show_cluster_timecourse(clusters, T_obs, stc, 561, n_vertices, n_times, hemi=hemi)


mean_175_220 = armenian_stc.copy().crop(tmin=0.175,tmax=0.220)
mean_175_220.data = mean_175_220.data.mean(axis=1).reshape(20484, 1)
tview_kwargs = dict(hemi='lh', clim=dict(kind='value', pos_lims=tpoint_lims), colormap=cmap, 
                time_unit='s', smoothing_steps=5, size = 900, time_viewer = True,
                view_layout='horizontal', cortex='bone', background=(1.,1.,1.,0.))


tind, vind = mean_175_220.get_peak(vert_as_index=True, time_as_index=True)
mean_175_220.data[tind,vind]

mean_175_220 = armenian_stc.copy().crop(tmin=0.175,tmax=0.220)
mean_175_220.data = mean_175_220.data.mean(axis=1).reshape(20484, 1)

# snapshots from time window 175-220
mean_images_175_220 = []
for v in views:
    l = rois
    if v == 'ventral':
        brain = brain_snapshot(mean_175_220, views=ventral_view, t=0.175, lims=tpoint_lims, roi=l)
    else:
        brain = brain_snapshot(mean_175_220, views=v, t=0.175, lims=tpoint_lims, roi=l)
    
    img = brain.screenshot(time_viewer=False, mode='rgba') 
    lx, ly,a = img.shape
    # Cropping 
    img = img[lx // c_u: - lx// c_l, :, :]
    
    mean_images_175_220.append(img)

mean_220_270 = armenian_stc.copy().crop(tmin=0.220,tmax=0.290)
mean_220_270.data = mean_220_270.data.mean(axis=1).reshape(20484, 1)
tind, vind = mean_220_270.get_peak(vert_as_index=True, time_as_index=True)
print(mean_220_270.data[tind,vind])

# snapshots from time window 220-270
mean_images_220_270 = []
for v in views:
    l = rois
    if v == 'ventral':
        brain = brain_snapshot(mean_220_270, views=ventral_view, t=0.220, lims=tpoint_lims, roi=l)
    else:
        brain = brain_snapshot(mean_220_270, views=v, t=0.220, lims=tpoint_lims, roi=l)
    
    img = brain.screenshot(time_viewer=False, mode='rgba') 
    lx, ly,a = img.shape
    # Cropping 
    img = img[lx // c_u: - lx// c_l, :, :]
    
    mean_images_220_270.append(img)


images = [mean_images_175_220, mean_images_220_270]

widths = [3, 3, 3]
heights = [3, 3]

figure_av = plt.figure(figsize=(17.5*cm,17.5*cm*2/3), dpi=142.12)
gs_tp = figure_av.add_gridspec(ncols=3, nrows=2, width_ratios=widths,
                          height_ratios=heights, left=0.03, right=0.90, bottom=0.2)

gs_tp.update(wspace=0.0, hspace=0.00) # set the spacing between axes. 
for col in range(4):
    for row, title in zip([0,1,2], ['175-220 ms', '220-290 ms']):
        if col < 3:
            axs1 = plt.subplot(gs_tp[row, col])
            im = axs1.imshow(images[row][col])
            axs1.set_axis_off()
            axs1.set_aspect('equal')
        if col==0:
            axs1.set_title(title, x=-0.05, y=0.30, rotation='vertical', fontsize=8)
    
figure_av.suptitle('Armenian Condition\nMean cluster t-value across time window', fontsize=8, fontweight='bold')

colorbar='v'
if colorbar == 'v':
    widths = [1]
    heights = [1, 1, 1, 1, 1,1,1,1,1]
    gs_cb = figure_av.add_gridspec(ncols=1, nrows=9, width_ratios=widths,
                                   height_ratios=heights, top=0.85,bottom=0.25,
                                   left=0.94, right=0.95)
    ax = plt.subplot(gs_cb[3:9, 0])
    cbar = mne.viz.plot_brain_colorbar(ax, clim=dict(kind='value', pos_lims=tpoint_lims), orientation='vertical',
                                label='t-value', bgcolor=bgc, colormap=cmap)
elif colorbar=='h':
    heights = [1]
    widths = [1, 1, 1, 1, 1,1,1,1,1]
    gs_cb = figure_av.add_gridspec(ncols=9, nrows=1, width_ratios=widths,
                                   height_ratios=heights, top=0.150,bottom=0.13,
                                   left=0.0, right=1)
    ax = plt.subplot(gs_cb[0, 2:6])
    cbar = mne.viz.plot_brain_colorbar(ax, clim=dict(kind='value', pos_lims=tpoint_lims), orientation='horizontal',
                                label='t-value', bgcolor=bgc, colormap=cmap)
    
cbar.ax.tick_params(labelsize=6, pad=1) 
cbar.set_ticks([-8,-6,-4,-2,0,2,4,6,8], update_ticks=True)
cbar.set_label(label= 't-value', size=6,weight='bold')


mpl.rcParams.update(new_rc_params)
figure_av.savefig('fg_armenian_average_'+ date+'_max-t.svg', figsize=(17.5*cm,17.5*cm*2/3), dpi=600)


# Normal - Armenian Difference

# use these lims for Normal-Armenian condition figures (peak t-val of difference)
tpoint_lims=[0.,2,4.83]


tmin = 0.175
tmax = 0.335
time_label = str(int(tmin * 1000)) + '-' + str(int(tmax * 1000)) + 'ms'
conditions = ['normal', 'armenian']
stc_file = op.join(data_folder, conditions[0], method, conditions[0] + '_average')
stc = mne.read_source_estimate(stc_file, 'fsaverage').crop(tmin,tmax)
stc.data = np.zeros(stc.data.shape)

result_file = op.join(result_folder, hemi,method, 'cluster_timecourse', 'normalized_contrast_' + conditions[0] + '_' + conditions[1] + time_label + '_nperm-' + str(int(n_permutations))+'_all_clu.pkl')
clu = pickle.load( open( result_file, "rb" ) )
T_obs, clusters, cluster_p_values, H0 = clu
n_times, n_vertices = T_obs.shape
data = np.zeros((n_vertices, n_times))
# Select clusters with a p value below 0.05
alpha = 0.05
show_cluster_p_vals(alpha, cluster_p_values)

brain, normal_armenian_stc = show_cluster_timecourse(clusters, T_obs, stc, 3, n_vertices, n_times, hemi=hemi)


mean_175_220 = normal_armenian_stc.copy().crop(tmin=0.175,tmax=0.220)
mean_175_220.data = mean_175_220.data.mean(axis=1).reshape(20484, 1)
tview_kwargs = dict(hemi='lh', clim=dict(kind='value', pos_lims=tpoint_lims), colormap=cmap, 
                time_unit='s', smoothing_steps=5, size = 900, time_viewer = True,
                view_layout='horizontal', cortex='bone', background=(1.,1.,1.,0.))

tind, vind = mean_175_220.get_peak(vert_as_index=True, time_as_index=True)
mean_175_220.data[tind,vind]


mean_175_220 = normal_armenian_stc.copy().crop(tmin=0.175,tmax=0.220)
mean_175_220.data = mean_175_220.data.mean(axis=1).reshape(20484, 1)

# snapshots from time window 175-220
mean_images_175_220 = []
for v in views:
    l = rois
    if v == 'ventral':
        brain = brain_snapshot(normal_armenian_stc, views=ventral_view, t=0.175, lims=tpoint_lims, roi=l)
    else:
        brain = brain_snapshot(normal_armenian_stc, views=v, t=0.175, lims=tpoint_lims, roi=l)
        
    img = brain.screenshot(time_viewer=False, mode='rgba') 
    lx, ly,a = img.shape
    # Cropping 
    img = img[lx // c_u: - lx// c_l, :, :]
    
    mean_images_175_220.append(img)


mean_220_270 = normal_armenian_stc.copy().crop(tmin=0.220,tmax=0.290)
mean_220_270.data = mean_220_270.data.mean(axis=1).reshape(20484, 1)
tind, vind = mean_220_270.get_peak(vert_as_index=True, time_as_index=True)
print(np.max(mean_220_270.data))

# snapshots from time window 220-270
mean_images_220_270 = []
for v in views:
    l = rois
    if v == 'ventral':
        brain = brain_snapshot(mean_220_270, views=ventral_view, t=0.220, lims=tpoint_lims, roi=l)
    else:
        brain = brain_snapshot(mean_220_270, views=v, t=0.220, lims=tpoint_lims, roi=l)
        
    img = brain.screenshot(time_viewer=False, mode='rgba') 
    lx, ly,a = img.shape
    # Cropping 
    img = img[lx // c_u: - lx// c_l, :, :]
    
    mean_images_220_270.append(img)



images = [mean_images_175_220, mean_images_220_270]

widths = [3, 3, 3]
heights = [3, 3]

figure_av = plt.figure(figsize=(18.0*cm,14.0*cm), dpi=142.12)
gs_tp = figure_av.add_gridspec(ncols=3, nrows=2, width_ratios=widths,
                          height_ratios=heights, left=0.03, right=0.97, bottom=0.151)

gs_tp.update(wspace=0.0, hspace=0.00) # set the spacing between axes. 
for col in range(4):
    for row, title in zip([0,1,2], ['175-220 ms', '220-290 ms']):
        if col < 3:
            axs1 = plt.subplot(gs_tp[row, col])
            im = axs1.imshow(images[row][col])
            axs1.set_axis_off()
            axs1.set_aspect('equal')
        if col==0:
            axs1.set_title(title, x=-0.05, y=0.30, rotation='vertical', fontsize=8)

    
figure_av.suptitle('Normal-Armenian Condition\nMean cluster t-value across time window', fontsize=8, fontweight='bold')

colorbar='h'
if colorbar == 'v':
    widths = [1]
    heights = [1, 1, 1, 1, 1,1,1,1,1]
    gs_cb = figure_av.add_gridspec(ncols=1, nrows=9, width_ratios=widths,
                                   height_ratios=heights, top=0.85,bottom=0.25,
                                   left=0.94, right=0.95)
    ax = plt.subplot(gs_cb[3:9, 0])
    cbar = mne.viz.plot_brain_colorbar(ax, clim=dict(kind='value', pos_lims=tpoint_lims), orientation='vertical',
                                label='t-value', bgcolor=bgc, colormap=cmap)
elif colorbar=='h':
    heights = [1]
    widths = [1, 1, 1, 1, 1,1,1,1,1]
    gs_cb = figure_av.add_gridspec(ncols=9, nrows=1, width_ratios=widths,
                                   height_ratios=heights, top=0.150,bottom=0.13,
                                   left=0.0, right=1)
    ax = plt.subplot(gs_cb[0, 2:6])
    cbar = mne.viz.plot_brain_colorbar(ax, clim=dict(kind='value', pos_lims=tpoint_lims), orientation='horizontal',
                                label='t-value', bgcolor=bgc, colormap=cmap)
    
cbar.ax.tick_params(labelsize=6, pad=1) 
cbar.set_ticks([-4,-3,-2,-1,0,1,2,3,4], update_ticks=True)
cbar.set_label(label= 't-value', size=6,weight='bold')


figure_av.savefig('fg_normal_armenian_average_'+date+'_max-t.svg', figsize=(17.5*cm,17.5*cm*2/3), dpi=600)



