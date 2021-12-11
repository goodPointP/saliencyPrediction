import torch.nn as nn
import os
import torch 
from PIL import Image

class VGG_homemade(nn.Module):
    """ VGG implementation to be in control of params, but otherwise completely based on the original architecture
    for original classifier use: 
        
        nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000)
            )
        
    """
    def __init__(self, features = None, classifier = None, preset = True):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        if preset:
            self.features = encoder()
            self.classifier = decoder()
        else:
            self.features = features
            self.classifier = classifier 
        
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        if self.classifier:
            x = self.classifier(x)
        return x



def encoder(layers = None,
            conv_params = None,
            pool_params = None,
            input_size = 3,
            relu=True,
            batch=True):
    
    if not layers:
        layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    if not conv_params:
        conv_params = [3, 1, 1]
    if not pool_params:
        pool_params = [2, 2, 0, 1, False]
    
    sequence = []
    input_size = input_size
    k, s, p = (*conv_params,)
    if not pool_params:
        MaxPool = nn.MaxPool2d(k, s, p)
    else:
        MaxPool = nn.MaxPool2d(*pool_params,)
        
    for layer in layers:
        if layer == 'M':
            sequence.append(MaxPool)
        else:
            if (relu == True) and (batch == True):
                sequence.append(nn.Conv2d(input_size, layer, k, s, p))
                sequence.append(nn.ReLU(inplace=True))
            else:
                sequence.append(nn.Conv2d(input_size, layer, k, s, p))
            input_size = layer
    return nn.Sequential(*sequence)


def decoder(layers = None,
            conv_params = None,
            upsampler = None,
            input_size = 512,
            ):
    
    if not layers:
        layers = ['IA', 'U', 'IB', 'U', 128, 'U', 64, 'U', 3, 'U', 1]
    if not conv_params:
        conv_params = [3, 1, 1]
    if not upsampler:
        upsampler = [2, "bicubic", False]
    
    
    sequence = []
    input_size = input_size
    k, s, p = (*conv_params,)
    scale, mode, align = (*upsampler,)
    upsample = nn.Upsample(scale_factor=scale, mode=mode, align_corners=align)
        
    for layer in layers:
        if layer == layers[-1]:
            sequence.append(nn.Conv2d(input_size, 1, 1))
        elif layer == 'U':
            sequence.append(upsample)
        elif layer == 'IA':
            sequence.append(inception_blockA())
            input_size = 512
        elif layer == 'IB':
            sequence.append(inception_blockB())
            input_size = 256
        else:
            sequence.append(standard_conv(input_size, layer, k, s, p))
            input_size = layer
    return nn.Sequential(*sequence)


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


class inception_blockA(nn.Module):
    """
    architecture borrowed from "https://github.com/pytorch/vision/blob/6a1d9ee7843e9edbf2c8a64b1517719a70b51f39/torchvision/models/inception.py#L398"
    """
    def __init__(self):
        super().__init__()
        conv_block = standard_conv
        self.branch1_1x1 = conv_block(512, 128, k=1, s=1)
        
        self.branch2_1x1 = conv_block(512, 128, k=1, s=1)
        self.branch2_3_3 = conv_block(128, 256, k=3, s=1, p=1)
        
        self.branch3_1x1 = conv_block(512, 32, k=1, s=1)
        self.branch3_3x3_2 = conv_block(32, 64, k=3, s= 1, p=1,d=2)
        
        self.branch4_3x3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_1x1 = conv_block(512, 64, k=1, s=1)
        
    def _forward(self, x):
        
        print(x.shape)
        b1 = self.branch1_1x1(x)
        print(b1.shape)
        
        b2 = self.branch2_1x1(x)
        print(b2.shape)
        b2 = self.branch2_3_3(b2)
        print(b2.shape)

        b3 = self.branch3_1x1(x)
        print(b3.shape)
        b3 = self.branch3_3x3_2(b3)
        print(b3.shape)

        b4 = self.branch4_3x3(x)
        print(b4.shape)
        b4 = self.branch4_1x1(b4)
        print(b4.shape)
        
        output = [b1, b2, b3, b4]
        
        return output
      
        
      
        
        # b1 = self.branch1_1x1(x)
        
        
        # b2 = self.branch2_1x1(x)
        # b2 = self.branch2_3_3(b2)
        
        # b3 = self.branch3_1x1(x)
        # b3 = self.branch3_3x3_2(b3)
        
        # b4 = self.branch4_3x3(x)
        # b4 = self.branch4_1x1(b4)
        
        # output = [b1, b2, b3, b4]
        
        # return output
    
    def forward(self, x):
        x = self._forward(x)
        return torch.cat(x, 1)
    
    
class inception_blockB(nn.Module):
    """
    architecture borrowed from "https://github.com/pytorch/vision/blob/6a1d9ee7843e9edbf2c8a64b1517719a70b51f39/torchvision/models/inception.py#L398"
    """
    def __init__(self):
        super().__init__()
        conv_block = standard_conv
        self.branch1_1x1 = conv_block(256, 64, k=1, s=1)
        
        self.branch2_1x1 = conv_block(256, 64, k=1, s=1)
        self.branch2_3_3 = conv_block(64, 128, k=3, s=1, p=1)
        
        self.branch3_1x1 = conv_block(256, 16, k=1, s=1)
        self.branch3_3x3_2 = conv_block(16, 32, k=3, s=1,p=1, d=2)
        
        self.branch4_3x3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_1x1 = conv_block(256, 32, k=1, s=1)
        
    def _forward(self, x):
        print(x.shape)
        b1 = self.branch1_1x1(x)
        print(b1.shape)
        
        b2 = self.branch2_1x1(x)
        print(b2.shape)
        b2 = self.branch2_3_3(b2)
        print(b2.shape)

        b3 = self.branch3_1x1(x)
        print(b3.shape)
        b3 = self.branch3_3x3_2(b3)
        print(b3.shape)

        b4 = self.branch4_3x3(x)
        print(b4.shape)
        b4 = self.branch4_1x1(b4)
        print(b4.shape)
        
        output = [b1, b2, b3, b4]
        
        return output
    
    def forward(self, x):
        x = self._forward(x)
        return torch.cat(x, 1)


class standard_conv(nn.Module): 
    """
    Based on https://github.com/MichiganCOG/TASED-Net/blob/master/model.py
    """
    def __init__(self, in_size=None, out_size=None, k=None, s=1, p=0, d=1):
        super(standard_conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_size, 
                              out_channels=out_size, 
                              kernel_size=k, 
                              stride=s, 
                              padding=p, 
                              bias=False,
                              )
        self.bn = nn.BatchNorm2d(out_size, eps=1e-3, momentum=0.001, affine=False) 
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x