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
    python3 inference.py -m models/newnet_model -o models/inferencetest -i 0 10
"""

import torch
import argparse
from utils_data import inference_pipe
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='model inference')
parser.add_argument('-m', default=None, required=True, dest='model')
parser.add_argument('-i_tuple', default=None, dest='input',type=int, nargs="+")
parser.add_argument('-i_data', default=None,  dest='input')
parser.add_argument('-o', default=None, required=True, dest='outfile')
parser.add_argument('-s', default=False, required=False, dest='save')
parser.add_argument('-so', default=None, required='-s' in sys.argv, dest='dataset_outfile')

args = parser.parse_args()

if not args.input:
    sys.exit("input not found. Use '-i tuple' or '-i data' with two ints or a dataloader, respectively")

accuracy = 0
heatmaps = []

# if not missing_keys:
print("loading model")
gazenet = torch.load(args.model)
gazenet.to(device)
gazenet.eval()

if len(args.input) > 1:
    print("constructing dataset")
    args.input = tuple(args.input)
    dataset = inference_pipe(args.input, batch_size = 1, workers = 10)
    if args.save:
        torch.save(dataset, args.dataset_outfile)
else:
    print("using pre-constructed dataloader")
    dataset = args.input

print("computing predictions")
for idx, (X, y) in enumerate(dataset):
    
    X = X.to(device)
    y = y.to(device).float()

    predict = gazenet(X)
    
    # loss = loss_fn(predict_test, y_test)
    # valid_loss+=loss.item()*X_test.size(0)
    # accuracy metric here
    heatmaps.append(predict.cpu())

torch.save(heatmaps, args.outfile)
print("accuracy: {}".format(accuracy))
# else:
#     print("inference cancelled. Missing keys: {}".format(missing_keys))