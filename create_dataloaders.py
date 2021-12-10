import h5py
import pandas as pd
from torchvision import transforms
import torch
from heatmap_creation import heatmapper
import CNN_functions
import numpy as np
#%%
collection = h5py.File('../../Datasets/nature_dataset/etdb_v1.0.hdf5')

#%%  access a given dataset 
# run collection.keys() to see datasets
# select a dataset with collection['key']

baseline = collection['Baseline'] #use baseline.keys() to see options

df_baseline = pd.DataFrame()
for key in baseline.keys():
    df_baseline[key] = pd.Series(baseline[key])
    
#%%

mappy = heatmapper(df_baseline, (1280, 960))

#%%
impaths = mappy.paths
targets = torch.tensor(np.load('heatmaps_full.npz')['heatmaps'])
#%% 
transformer = transforms.Compose([transforms.Resize(256), 
                            transforms.CenterCrop(224), 
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

   
samples = len(impaths)
bs = 16
workers = 2

gazedata_train = CNN_functions.gazedataset(impaths[:int(samples/5*4)], 
                                           targets[:int(samples/5*4)], 
                                           transform=transformer,
                                           )

train_loader = torch.utils.data.DataLoader(gazedata_train, 
                                           batch_size=bs, 
                                           shuffle=True, 
                                           num_workers=workers
                                           )


gazedata_test = CNN_functions.gazedataset(impaths[int(samples/5*4):int(samples)], 
                                          targets[int(samples/5*4):int(samples)], 
                                          transform=transformer,
                                          )

test_loader = torch.utils.data.DataLoader(gazedata_test, 
                                          batch_size=bs, 
                                          shuffle=True, 
                                          num_workers=workers
                                          )

torch.save(test_loader,'test_loader.pt')
torch.save(train_loader, 'train_loader.pt')