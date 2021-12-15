from torchvision import models, transforms
import torch
import PIL
import torch.nn as nn
import torch.functional as F
import CNN_functions
import torch.optim as optim

#%% Instantiate original VGG model WITH weights
vgg_base = models.vgg16(pretrained=True)

#%% Create encoder and decoder for custom model

features = CNN_functions.encoder()
classifier = CNN_functions.decoder()
    
gazenet = CNN_functions.VGG_homemade(features, preset=False)

#%% Initializing weights BEFORE adding new classifier
base_state_dict = vgg_base.state_dict()
gazenet.load_state_dict(base_state_dict, strict=False)

#%% Disable parameter tuning for original weights
# for param in gazenet.parameters():
#     param.requires_grad = False
#%% Add fully convolutional classifier

gazenet.classifier = classifier

#%%


loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.414]))
optimizer = optim.SGD(gazenet.parameters(), lr=0.01, momentum=0.9,weight_decay=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3)
epoch = 0
loss = 0
torch.save({
            'epoch': epoch,
            'model_state_dict': gazenet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'scheduler': scheduler.state_dict(),
            }, 'models/gazenet')
