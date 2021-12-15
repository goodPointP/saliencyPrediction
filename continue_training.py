import torch
import os
import utils_nn
import torch.optim as optim
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='load model to continue training')
parser.add_argument('-model', default=None, dest='model')


args = parser.parse_args()

if args.model is not None:
    
    gazenet = utils_nn.VGG_homemade()
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.1],device=device), reduction='mean')
    optimizer = optim.SGD(gazenet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    
    gazenet, optimizer, scheduler, start_epoch, loss = utils_nn.load_checkpoint(gazenet, 
                                                                                optimizer, 
                                                                                scheduler, 
                                                                                args.model)

    print("model loaded")
else:
    print("no model supplied")
