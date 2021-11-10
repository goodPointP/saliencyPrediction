import h5py
import pandas as pd
import custom
import numpy as np
import time


#%% #Collection of datasets

collection = h5py.File('../../Datasets/nature_dataset/etdb_v1.0.hdf5')

#%%  access a given dataset 
# run collection.keys() to see datasets
# select a dataset with collection['key']

baseline = collection['Baseline'] #use baseline.keys() to see options

df_baseline = pd.DataFrame()
for key in baseline.keys():
    df_baseline[key] = pd.Series(baseline[key])
    

#%%
subject_index = 40
trial_number = 3
name_of_experiment = 'Baseline'

f, hm = custom.compute_heatmap(df = df_baseline, s_index = subject_index, s_trial = trial_number, experiment = name_of_experiment)
#%% feature engineering

df_engineered = custom.f_engi(df_baseline)
#%% feature extraction

relevant_columns = ['pupil','x','y', 'time']
df_extracted = custom.f_extraction(df_engineered, relevant_columns)

#%%
 
su, tr = custom.get_subject_trial(df_baseline)

#%% 500 images = 30sec
t0 = time.time()
tester = custom.compute_heatmap(df = df_baseline, s_index = su[:2], s_trial=tr, draw=False, experiment = 'Baseline')
t1 = time.time()
print(t1-t0)

#%% Mock examples
"""
notes: normalize pixel values for efficiency?
different shape&size input?
different shape&size output?
very carefully consider the gaussian function (and its params) in use. Right now just copy pasta, but has a lot of impact.
What kind of accuracy/loss criterion?

uncomment lines below to test compute_heatmap() in a ... not functional way
"""
# dims = (1280, 960)
# s_index = 45
# trial = 4

# filedata = df.loc[(df.SUBJECTINDEX == s_index) & (df.trial == trial)]

# cat = int(filedata.category.iloc[0])
# filenr = int(filedata.filenumber.iloc[0])

# _, ext = os.path.splitext(os.listdir('../../Datasets/nature_dataset/{}/'.format(cat))[-1]) #Yes, I know this isn't exactly codeporn
# imgpath = '../../Datasets/nature_dataset/{}/{}{}'.format(cat, filenr, ext)
# img = Image.open('../../Datasets/nature_dataset/{}/{}{}'.format(cat, filenr, ext))


# fixations = np.array((filedata.start, filedata.end, np.abs(filedata.start-filedata.end), filedata.x, filedata.y)).T

# draw_heatmap(fixations, dims, imagefile=imgpath)

