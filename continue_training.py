"""
To run : <env> <python file> < -m model_dict_path> < -o output_path> < -tr traindata_path> < -ts testdata_path> < -e epochs>
    
e.g. : python3 continue_training.py -m models/gazenet_post_train_100_epochs -o models/newnet -tr train_loader.pt -ts test_loader.pt -e 3

"""

import torch
import utils_nn
import torch.optim as optim
import argparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='load model to continue training')
parser.add_argument('-m', default=None, dest='model')
parser.add_argument('-o', default=None, dest='outfile')
parser.add_argument('-tr', default=None, dest='traindata')
parser.add_argument('-ts', default=None, dest='testdata')
parser.add_argument('-e', default=None, dest='epochs', type=int)

args = parser.parse_args()

#%%

if args.model is not None:
    
    gazenet = utils_nn.VGG_homemade()
    gazenet.to(device)
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

train_losses=[]
valid_losses=[]



if (args.outfile is not None) and (args.traindata is not None) and (args.testdata is not None) and (args.epochs is not None):
    #train loop here
    # gazenet.to(device)
    train_loader = torch.load(args.traindata)
    test_loader = torch.load(args.testdata)
    epochs = args.epochs
    
    if __name__ == '__main__':
        for epoch in range(0+start_epoch, epochs+start_epoch):
            
            print("starting epoch: {} with learning rate: {}".format(epoch, optimizer.param_groups[0]['lr']))
            
            train_loss=0.0
            valid_loss=0.0
            gazenet.train()
            
            for idx, (X, y) in enumerate(train_loader):
        
                X = X.to(device)
                y = y.to(device).float()
                
                optimizer.zero_grad()
                predict = gazenet(X)
                            
                loss=loss_fn(predict,y)
                loss.backward()
                
                train_loss+=loss.item()*X.size(0)
                optimizer.step()

                torch.cuda.empty_cache()
                
            with torch.no_grad():
                gazenet.eval()
                
                for idx_t, (X_test, y_test) in enumerate(test_loader):
                        
                    X_test = X_test.to(device)
                    y_test = y_test.to(device).float()
            
                    predict_test = gazenet(X_test)
                    
                    loss = loss_fn(predict_test, y_test)
                    valid_loss+=loss.item()*X_test.size(0)
                    
            scheduler.step(valid_loss)
            
            
            train_loss=train_loss/len(train_loader.sampler) 
            valid_loss=valid_loss/len(test_loader.sampler)
            print("loss in training: {}, loss in validation: {}".format(train_loss, valid_loss))
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
    
    
    torch.save({
                'epoch': epoch,
                'model_state_dict': gazenet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': loss,
                }, args.outfile+"_dict"
        )

    torch.save(gazenet, args.outfile+"_model")
    
else:
    for key, value in vars(args).items():
        if value is None:
            print(key, "not supplied")
    
    
    
    