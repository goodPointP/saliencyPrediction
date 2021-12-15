
import torch
from heatmapper import heatmapper
import utils_data
#%%

df_baseline, dims = utils_data.baseline_dset()
mappy = heatmapper(df_baseline, dims)

impaths = mappy.paths
targets = mappy.compute()

#%% 

train_loader, test_loader = utils_data.loader_pipe(impaths, targets, batch_size=16)

#%%

torch.save(test_loader,'test_loader.pt')
torch.save(train_loader, 'train_loader.pt')