import h5py
import pandas as pd
import gaze_functions
import torch
import numpy as np
from torchvision import transforms
import torch.optim as optim
import CNN_functions
from PIL import Image
import matplotlib.pyplot as plt
from heatmap_creation import heatmapper
#%%


checkpoint = torch.load('models/gazenet')
gazenet = CNN_functions.VGG_homemade()
gazenet.load_state_dict(checkpoint['model_state_dict'])
for param in gazenet.features.parameters():
    param.requires_grad = False



#%% #Collection of datasets
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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


#%%

gazenet.to(device)

loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.414],device=device))
optimizer = optim.SGD(gazenet.parameters(), lr=0.01, momentum=0.9,weight_decay=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)
train_losses=[]
valid_losses=[]


epochs = 2
if __name__ == '__main__':
    for epoch in range(0, epochs):
        print("starting epoch: {}".format(epoch))
        train_loss=0.0
        valid_loss=0.0
        gazenet.train()
        for idx, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)
            # print("here")
            optimizer.zero_grad()
            # print("there")
            predict = gazenet(X)   
            # print("get")
            loss=loss_fn(predict,y)
            # print("jefe")
            loss.backward()
            
            scheduler.step()
            train_loss+=loss.item()*X.size(0)
            torch.cuda.empty_cache()
            # if idx > 10:
            #     print("stopping train after batch: ", idx)
            #     break   
        with torch.no_grad():
            gazenet.eval()
            for idx_t, (X_test, y_test) in enumerate(test_loader):
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                # if idx_t < 2:
                predict_test = gazenet(X_test)
                loss = loss_fn(predict_test, y_test)
                valid_loss+=loss.item()*X_test.size(0)
                # else:
                #     print("stopping eval")
                #     break
        train_loss=train_loss/len(train_loader.sampler) 
        valid_loss=valid_loss/len(test_loader.sampler)
        print("loss in training: {}, loss in validation: {}".format(train_loss, valid_loss))
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        
print("done")
#%%

print("training losses from start to end: {}, validation losses from start to end: {}".format(train_losses, valid_losses))
# print(predict_test.cpu().detach().numpy()[0])


#%% 500 images = 30sec

torch.save({
            'epoch': epoch,
            'model_state_dict': gazenet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': loss,
            }, "models/gazenet_post_train"
    )


#%%

# subject_index = 1
# trial_number = 2
# name_of_experiment = 'Baseline'

# f, hm = gaze_functions.compute_heatmap(df = df_baseline, s_index = subject_index, s_trial = trial_number, experiment = name_of_experiment)



# t0 = time.time()
# tester = gaze_functions.compute_heatmap(df = df_baseline, s_index = su[:2], s_trial=tr, draw=False, experiment = 'Baseline')
# t1 = time.time()
# print(t1-t0)


