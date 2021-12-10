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

cfg_encode = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
convolution_settings = [3, 1, 1]
pool_settings = [2, 2, 0, 1, False]

features = CNN_functions.encoder(cfg_encode, convolution_settings, pool_settings, 3)
#%%


cfg_decode = [512, 'U', 256, 'U', 128, 'U', 64, 'U', 3, 'U', 1]
cfg_up = [2, "bicubic", False]

classifier = CNN_functions.decoder(layers=cfg_decode, conv_params=convolution_settings, upsampler=cfg_up, input_size=512, relu=True)
    
gazenet = CNN_functions.VGG_homemade(features, preset=False)

#%% Initializing weights BEFORE adding new classifier
base_state_dict = vgg_base.state_dict()
gazenet.load_state_dict(base_state_dict, strict=False)

#%% Disable parameter tuning for original weights
for param in gazenet.parameters():
    param.requires_grad = False
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


# #%%
# checkpoint = torch.load('models/VGG_custom')
# #%% Transformer to conform with VGG requirements

# trans = transforms.Compose([transforms.Resize(256), 
#                             transforms.CenterCrop(224), 
#                             transforms.ToTensor(), 
#                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# #%% Get sample for test
# input_image = PIL.Image.open('C:/Users/Sebastian/Desktop/CS3/Datasets/nature_dataset/8/5.png')
# input_tensor = trans(input_image)
# input_batch = input_tensor.unsqueeze(0) #make 4D (req by torch models) [samples, dims, h, w]

# #%% Testing

# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     vgg_base.to('cuda')
#     VGG_custom.to('cuda')
    
# with torch.no_grad():
#     output_base = vgg_base(input_batch)
#     output_custom = VGG_custom(input_batch)

#%% Outputs

# proba = torch.nn.functional.softmax(output_base[0], dim=0)
# proba_custom = torch.nn.functional.softmax(output_custom[0], dim=0)












