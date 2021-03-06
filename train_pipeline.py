import torch
import torch.optim as optim
import utils_nn

#%%


checkpoint = torch.load('models/gazenet_dict')
gazenet = utils_nn.VGG_homemade()
gazenet.load_state_dict(checkpoint['model_state_dict'])
# for param in gazenet.features.parameters():
#     param.requires_grad = False


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


train_loader = torch.load('train_loader.pt')
test_loader = torch.load('test_loader.pt')


#%%

gazenet.to(device)

loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.1],device=device), reduction='mean')
optimizer = optim.SGD(gazenet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
train_losses=[]
valid_losses=[]


epochs = 100
if __name__ == '__main__':
    for epoch in range(0, epochs):
        
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
                    }, "models/gazenet_mid_train_dict"
            )

torch.save(predict_test, 'sample_heatmap_prediction_idx:{}.pt'.format(idx_t))
print("done")
#%%

print("training losses from start to end: {}, validation losses from start to end: {}".format(train_losses, valid_losses))


#%% 

torch.save({
            'epoch': epoch,
            'model_state_dict': gazenet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': loss,
            }, "models/gazenet_post_train_{}_epochs_dict".format(epochs)
    )


