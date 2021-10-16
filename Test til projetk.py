import numpy as np
import h5py
import cv2
from zipfile import ZipFile
import pandas as pd

#%% #Collection of datasets
collection = h5py.File('C:/Users/Sebastian/Desktop/nature_dataset/etdb_v1.0.hdf5')
#%%  access a given dataset 
# run collection.keys() to see datasets
# select a dataset with collection['key']

baseline = collection['Baseline'] #use baseline.keys() to see options
entry = 90000
#retrive category and filenumber:
cat, filenr = int(baseline['category'][entry]), int(baseline['filenumber'][entry])
#%% open stimuli 
# stimuli = cv2.imread('C:/Users/Sebastian/Desktop/nature_dataset/Stimuli_{}.zip/{}/{}.png'.format(cat, cat, filenr), 0)
folder  = ZipFile('C:/Users/Sebastian/Desktop/nature_dataset/Stimuli_{}.zip'.format(cat), 'r')
imgbuffer = folder.read('{}/{}.png'.format(cat, filenr))
imgarr = np.frombuffer(imgbuffer, np.uint8)
img_np = cv2.imdecode(imgarr, cv2.IMREAD_COLOR)
cv2.imshow("test", img_np)
cv2.waitKey(0)

#Using ZipFile.extractall() is a scalable alternative
#%%
df = pd.DataFrame()
for key in baseline.keys():
    g2 = pd.Series(baseline[key])
    df[key] = pd.Series(baseline[key])

#%%
smalldf = df[:100]
