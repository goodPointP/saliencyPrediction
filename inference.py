""" 
    REQUIRED:
<env> 
< inference.py> 
< -m path/to_model> 
< -o path/to_outfile>
< i_tuple start_int end_int> OR < i_data path/to_dataloader>
    OPTIONALS: 
< -s boolean> : save dataloader created - requires "i_tuple"
< -so path/to_dataloader_outfile>
e.g. : 
    python3 inference.py -m models/newnet_model -o models/inferencetest -i_tuple 0 10
    python inference.py -m models/newnet_model -i_tuple 0 10 -so testPictures/testdataset
    
    
"""
from scipy.stats import wasserstein_distance
import torch
import numpy as np
import argparse
from utils_data import inference_pipe
import sys
from torchmetrics.functional import auroc, pearson_corrcoef, dice_score


#%%
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='model inference')
parser.add_argument('-m', default=None, required=False, dest='model')
parser.add_argument('-i_tuple', default=None, dest='tuple',type=int, nargs="+")
parser.add_argument('-i_data', default=None,  dest='dataset')
parser.add_argument('-i', default=None, dest='input')
parser.add_argument('-o', default=None, required=False, dest='outfile')
parser.add_argument('-so', default=None, required=False, dest='save')

args = parser.parse_args()

#%%

args.model = 'models/newnet_model'
args.tuple = 0,10
args.save = 'testPictures/testdataset'




# if not args.input:
#     sys.exit("input not found. Use '-i tuple' or '-i data' with two ints or a dataloader, respectively")

accuracy = 0
heatmaps = []


print("loading model")
gazenet = torch.load(args.model)
gazenet.to(device)
gazenet.eval()
cc = 0
auc = 0
emd = 0
if args.tuple:
    print(args.tuple)
    args.input = args.tuple
    print("constructing dataset")
    args.input = tuple(args.input)
    dataset = inference_pipe(args.input, batch_size = 1, workers=0)
    if args.save:
        torch.save(dataset, args.save)

elif args.dataset:
    print(args.dataset)
    args.input = args.dataset
    print("using pre-constructed dataloader")
    dataset = torch.load(args.input)
    
else:
    print(args.input)
    sys.exit("input not found. Use '-i tuple' or '-i data' with two ints or a dataloader, respectively")
#%%
if __name__ == '__main__':
    with torch.no_grad():
        print("computing predictions")
        for idx, (X, y) in enumerate(dataset):
            if len(X) == 3:
                X = X.unsqueeze(0)
                
            X = X.to(device)
            y = y.to(device).float()
        
            predict = gazenet(X)
            
            flat_p = predict.flatten()
            flat_y = y.flatten()
            
            cc += pearson_corrcoef(flat_p, flat_y).cpu().numpy() # 1 = good | 0 = bad
            auc += auroc(flat_p, flat_y.int(), pos_label=1).cpu().numpy() #1 = good | 0 = bad
            emd += wasserstein_distance(flat_p.cpu(), flat_y.cpu())#small = good
            
            heatmaps.append(predict.cpu())

accs = np.array((cc, auc, emd))/len(dataset.sampler)
if args.outfile:
    torch.save(heatmaps, args.outfile)
print("accuracy: {}".format(accs))