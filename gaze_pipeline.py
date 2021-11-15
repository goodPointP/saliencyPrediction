import h5py
import pandas as pd
import gaze_functions
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import nonechucks as nc
import torch.optim as optim

#%%
class gazenet(torch.nn.Module):
    def __init__(self):
        super(gazenet, self).__init__()
        
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.conv5 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.conv6 = torch.nn.Conv2d(256, 1, 1)
        
        self.upsample = torch.nn.Upsample(scale_factor=8, mode='bicubic', align_corners=False)
    
    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.pool(self.relu(self.conv4(x))) 
        # x = self.relu(self.conv5(x))
        x = self.pool(self.relu(self.conv6(x)))
        x = self.upsample(x)
        return x
    
#%%
class gazedataset(torch.utils.data.Dataset):
    def __init__(self, root, labels, transform=None):
        self.labels = labels
        self.root = root
        self.transform = transform

        
    def __dims__(self):
        return()
    
    def __len__(self):
        return(len(self.labels))
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        imgpath = self.root[idx]
        image = Image.open(imgpath)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label).type(image.type())
        return image, label
   
#%% #Collection of datasets
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
collection = h5py.File('../../Datasets/nature_dataset/etdb_v1.0.hdf5')

#%%  access a given dataset 
# run collection.keys() to see datasets
# select a dataset with collection['key']

baseline = collection['Baseline'] #use baseline.keys() to see options

df_baseline = pd.DataFrame()
for key in baseline.keys():
    df_baseline[key] = pd.Series(baseline[key])
    
#%%
# subject_index = 1
# trial_number = 2
# name_of_experiment = 'Baseline'

# f, hm = gaze_functions.compute_heatmap(df = df_baseline, s_index = subject_index, s_trial = trial_number, experiment = name_of_experiment)
#%%

su, tr = gaze_functions.get_subject_trial(df_baseline)
impaths = gaze_functions.image_paths(df_baseline, su, tr, 1)
heatmaps = gaze_functions.compute_heatmap(df = df_baseline, s_index = su, s_trial = tr, experiment = 'Baseline', last_tr=10, draw=False)

#%% 
transformer = transforms.Compose([transforms.ToTensor()])    

targets_scaled = (heatmaps-np.min(heatmaps))* (1/(np.max(heatmaps)-np.min(heatmaps)))
targets = targets_scaled


gazedata_train = gazedataset(impaths[:400], targets[:400], transform=transformer)
safe_gaze_train = nc.SafeDataset(gazedata_train) #Removes incomplete samples
train_loader = torch.utils.data.DataLoader(safe_gaze_train.dataset, batch_size=2, shuffle=False, num_workers=2)


gazedata_test = gazedataset(impaths[400:], targets[400:], transform=transformer)
safe_gaze_test = nc.SafeDataset(gazedata_test)
test_loader = torch.utils.data.DataLoader(safe_gaze_test.dataset, batch_size=2, shuffle=False, num_workers=2)


#%%
gnet = gazenet()
gnet = gnet.cuda()
loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(gnet.parameters(),lr=0.001)

train_losses=[]
valid_losses=[]


epochs = 1
if __name__ == '__main__':
    for epoch in range(epochs):
        train_loss=0.0
        valid_loss=0.0
        gnet.train()
        for idx, (X, y) in enumerate(train_loader):
            print("training batch: ", idx)
            X = X.to(device)
            y = y.to(device)
            # print("here")
            optimizer.zero_grad()
            # print("there")
            predict = gnet(X)   
            # print("get")
            loss=loss_fn(predict,y)
            # print("jefe")
            loss.backward()
            
            optimizer.step()
            train_loss+=loss.item()*X.size(0)
            torch.cuda.empty_cache()
            # if idx > 10:
            #     print("stopping train after batch: ", idx)
            #     break   
        with torch.no_grad():
            gnet.eval()
            for idx_t, (X_test, y_test) in enumerate(test_loader):
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                # if idx_t < 2:
                print("testing batch: ", idx_t)
                predict_test = gnet(X_test)
                loss = loss_fn(predict_test, y_test)
                valid_loss+=loss.item()*X_test.size(0)
                # else:
                #     print("stopping eval")
                #     break
        train_loss=train_loss/len(train_loader.sampler) 
        valid_loss=valid_loss/len(test_loader.sampler)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
print("done")
#%%

# preview = predict_test.cpu().detach().numpy()[0]

#%% 500 images = 30sec


# t0 = time.time()
# tester = gaze_functions.compute_heatmap(df = df_baseline, s_index = su[:2], s_trial=tr, draw=False, experiment = 'Baseline')
# t1 = time.time()
# print(t1-t0)

#%% Mock examples
"""
notes: normalize pixel values for efficiency?
    - test this
    
different shape&size input?
    - 
    
different shape&size output?
    - Deepfix paper read 
    
very carefully consider the gaussian function (and its params) in use. Right now just copy pasta, but has a lot of impact.
    - 
    
What kind of accuracy/loss criterion?
    - 


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

