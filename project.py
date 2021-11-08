import h5py
import pandas as pd
import custom
#%% #Collection of datasets

collection = h5py.File('../../Datasets/nature_dataset/etdb_v1.0.hdf5')
meta = pd.read_csv('../../Datasets/nature_dataset/meta.csv', sep='; ', engine='python')
# meta
#%%  access a given dataset 
# run collection.keys() to see datasets
# select a dataset with collection['key']

baseline = collection['Baseline'] #use baseline.keys() to see options

df_baseline = pd.DataFrame()
for key in baseline.keys():
    df_baseline[key] = pd.Series(baseline[key])
    

#%%
subject_index = 8
trial_number = 3
name_of_experiment = 'Baseline'

f = custom.compute_heatmap(df = df_baseline, s_index = subject_index, trial = trial_number, experiment = name_of_experiment)


#%% Mock examples
"""
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

