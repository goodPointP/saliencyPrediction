from torchvision import models
import torch
import utils_nn
import torch.optim as optim
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#%% Instantiate original VGG model WITH weights
vgg_base = models.vgg16(pretrained=True)


features = utils_nn.encoder()
classifier = utils_nn.decoder()
    
gazenet = utils_nn.VGG_homemade(features, preset=False)

#%% Initializing weights BEFORE adding new classifier
base_state_dict = vgg_base.state_dict()
gazenet.load_state_dict(base_state_dict, strict=False)

#%% Disable parameter tuning for original weights
# for param in gazenet.parameters():
#     param.requires_grad = False
#%% Add fully convolutional classifier

gazenet.classifier = classifier

#%%


loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.1],device=device), reduction='mean')
optimizer = optim.SGD(gazenet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
epoch = 0
loss = 0
torch.save({
            'epoch': epoch,
            'model_state_dict': gazenet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'scheduler': scheduler.state_dict(),
            }, 'models/gazenet_dict')
