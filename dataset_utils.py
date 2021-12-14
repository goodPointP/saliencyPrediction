import torch
from PIL import Image
from torchvision import transforms
import h5py
import pandas as pd
#%%

def baseline_dset():
    
    screen_dims = (1280, 960)
    collection = h5py.File('../../Datasets/nature_dataset/etdb_v1.0.hdf5')

    baseline = collection['Baseline'] #use baseline.keys() to see options

    df_baseline = pd.DataFrame()
    for key in baseline.keys():
        df_baseline[key] = pd.Series(baseline[key])

    return df_baseline, screen_dims

def loader_pipe(impaths, targets, parts=5, batch_size = 32, workers = 10):
    
    split = int(len(impaths)/(parts*(parts-1)))
    

    transformer = transforms.Compose([transforms.Resize(256), 
                                transforms.CenterCrop(224), 
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])
    
    dataset_train = gazedataset(impaths[:split],
                          targets[:split],
                          transform=transformer)
    
    dataset_test = gazedataset(impaths[split:],
                          targets[split:],
                          transform=transformer)
    
    train_loaded = torch.utils.data.DataLoader(dataset_train,
                                                 batch_size = batch_size,
                                                 shuffle = True,
                                                 num_workers=workers)
    
    test_loaded = torch.utils.data.DataLoader(dataset_test,
                                                 batch_size = batch_size,
                                                 shuffle = True,
                                                 num_workers=workers)
    
    return train_loaded, test_loaded


class gazedataset(torch.utils.data.Dataset):
    def __init__(self, root, labels, transform=None, target_transform=None):
        self.labels = labels
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return(len(self.labels))
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        imgpath = self.root[idx]
        image = Image.open(imgpath)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label).type(image.type())
        return image, label