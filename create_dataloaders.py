import h5py
import pandas as pd
from torchvision import transforms
import torch
from heatmap_creation import heatmapper
import CNN_functions
import numpy as np
import dataset_utils
#%%

df_baseline, dims = dataset_utils.baseline_dset()
mappy = heatmapper(df_baseline, dims)

impaths = mappy.paths
targets = mappy.compute()

#%% 

train_loader, test_loader = dataset_utils.loader_pipe(impaths, targets, batch_size=16)

#%%

torch.save(test_loader,'test_loader.pt')
torch.save(train_loader, 'train_loader.pt')