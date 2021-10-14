# DysControl_SourceLocalization

Source Localization analysis for the DysControl study. 

Notes:
In addition to the data, the following source spaces are needed for reproducing the analysis: 
- fsverage (https://surfer.nmr.mgh.harvard.edu/fswiki/FsAverage)
- fsaverage_sym (https://surfer.nmr.mgh.harvard.edu/fswiki/Xhemi).

fsaverage was obtained using the mne.datasets.fetch_fsaverage() command. fsaverage_sym was downloaded from Freesurfer's website. A BEM model for fsaverage_sym was created with mne.make_bem_model(). 

