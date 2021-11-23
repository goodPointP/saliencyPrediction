import h5py
import pandas as pd
import gaze_functions
import numpy as np
#%%
collection = h5py.File('../../Datasets/nature_dataset/etdb_v1.0.hdf5')
baseline = collection['Baseline'] #use baseline.keys() to see options

df_baseline = pd.DataFrame()
for key in baseline.keys():
    df_baseline[key] = pd.Series(baseline[key])
    
#%%
tr_pr_su = None

su, tr = gaze_functions.get_subject_trial(df_baseline)

impaths = gaze_functions.image_paths(df_baseline, 
                                     su, 
                                     tr, 
                                     last_tr = tr_pr_su)

heatmaps = gaze_functions.compute_heatmap(df = df_baseline, 
                                          s_index = su, 
                                          s_trial = tr, 
                                          experiment = 'Baseline', 
                                          last_tr= tr_pr_su, 
                                          draw=False)

#%%

impaths, heatmaps = gaze_functions.remove_invalid_paths(impaths, heatmaps)

np.save('impaths.npy', impaths)
np.save('heatmaps.npy', heatmaps)

