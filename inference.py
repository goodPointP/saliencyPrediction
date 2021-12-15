import torch
import utils_nn
import torch.optim as optim
import argparse
from utils_data import inference_pipe
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='model inference')
parser.add_argument('-m', default=None, required=True, dest='model')
parser.add_argument('-o', default=None, required=True, dest='outfile')
parser.add_argument('-i', default=None, required=True, dest='input')
parser.add_argument('-s', default=False, required=False, dest='save')
parser.add_argument('-so', default=None, required='-s' in sys.argv, dest='dataset_outfile')
# parser.add_argument('-t', default=None, dest='ground_truth', type=bool)

args = parser.parse_args([])

accuracy = 0
heatmaps = []

# if not missing_keys:
print("loading model")
gazenet = torch.load(args.model)
gazenet.to(device)
gazenet.eval()

print("constructing dataset")
if type(args.input) == tuple:
    dataset = inference_pipe(args.input, batch_size = 1, workers = 10)
    if args.save:
        torch.save(dataset, args.dataset_outfile)
else:
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