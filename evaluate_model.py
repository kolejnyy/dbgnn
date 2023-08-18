import numpy as np
import torch
from tqdm import tqdm

from torch_geometric.nn import GCN

from models.kpath_gnn import kPathGNN
from models.dbgnn import DBGNNModel
from models.gcn import GCNModel

from datasets.sddataset.sddataset import SDDataset
from datasets.wddataset.wddataset import WDDataset
from datasets.island.island import IslandDataset

from util.visualisation import *
from util.training import train


# dataset = SDDataset(load_path='datasets/sddataset/k5_g1000/', k_max=5, rho=np.array([1,2,4,8,16,128,16,8,4,2,1]), num_graphs=1000, avg_num_nodes=100, avg_degree=4)
dataset = IslandDataset(load_path='datasets/island/30x30_g1000/')
train_split = 800
valid_split = 900
test_split  = 1000

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device: ", device)

model = GCNModel(5, 10, 32, device)
# model = DBGNNModel(GCN(10, 32, 2, 32, norm = 'batch'),
						# GCN(32, 32, 2, 1, norm = 'batch'),
						# GCN(32, 32, 2, 1, norm = 'batch'),
						# device=device)
evaluations = 10

accuracies = np.zeros(evaluations)
precisions = np.zeros(evaluations)
recalls = np.zeros(evaluations)
f1s = np.zeros(evaluations)

for i in tqdm(range(evaluations)):
	_, _, acc, prec, rec, f1 = train(model, dataset, 100, 12, 0.003, train_split, valid_split, test_split, print_interval=10, valid_interval=5, db=False, thr_sv=False)
	accuracies[i] = acc
	precisions[i] = prec
	recalls[i] = rec
	f1s[i] = f1
    
# Print out the mean and standard deviation of the accuracies
print("Mean accuracy: ", np.mean(accuracies))
print("Accuracy SD: ", np.std(accuracies))

print("Mean precision: ", np.mean(precisions))
print("Precision SD: ", np.std(precisions))

print("Mean recall: ", np.mean(recalls))
print("Recall SD: ", np.std(recalls))

print("Mean F1: ", np.mean(f1s))
print("F1 SD: ", np.std(f1s))