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
from torch_geometric.datasets.twitch import Twitch
from torch_geometric.datasets import Planetoid

from util.visualisation import *
from util.training import train
from util.thresholding import generate_threshold_map

# dataset = SDDataset(load_path='datasets/sddataset/k5_g1000/', k_max=5, rho=np.array([1,2,4,8,16,128,16,8,4,2,1]), num_graphs=1000, avg_num_nodes=100, avg_degree=4)
# dataset = IslandDataset(load_path='datasets/island/30x30_g1000/')
# dataset = Twitch(root='datasets/twitch/', name='DE', transform=None)
dataset = Planetoid(root='datasets/', name='Pubmed', transform=None)

# train_split = 800
# valid_split = 900
# test_split  = 1000


graph = dataset[0]
test_split = np.arange(len(graph.x) // 10)*10
valid_split = np.arange((len(graph.x)-1)//10)*10+1
train_split = np.ones(len(graph.x), dtype=bool)
train_split[test_split] = False
train_split[valid_split] = False
y = graph.y
dataset[0].y[y == 2] = 1
dataset[0].y = 1-dataset[0].y


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device: ", device)

model = GCNModel(5, 500, 256, device)
# model = DBGNNModel(GCN(128, 128, 3, 128, norm = 'batch'),
# 						GCN(128, 128, 2, 1, norm = 'batch'),
# 						GCN(128, 128, 2, 1, norm = 'batch'),
# 						device=device)
evaluations = 50

accuracies = np.zeros(evaluations)
precisions = np.zeros(evaluations)
recalls = np.zeros(evaluations)
f1s = np.zeros(evaluations)

for i in tqdm(range(evaluations)):
	_, _, acc, prec, rec, f1 = train(model, dataset, 200, 32, 0.007,
				  					train_split, valid_split, test_split,
									print_interval=10, valid_interval=5,
				  					db=False, thr_sv=False,
									single_graph=True)
	accuracies[i] = acc
	precisions[i] = prec
	recalls[i] = rec
	f1s[i] = f1
    
# Print out the mean and standard deviation of the accuracies
print("Mean accuracy: ", np.mean(accuracies)*100)
print("Accuracy SD: ", np.std(accuracies)*100)

print("Mean precision: ", np.mean(precisions)*100)
print("Precision SD: ", np.std(precisions)*100)

print("Mean recall: ", np.mean(recalls)*100)
print("Recall SD: ", np.std(recalls)*100)

print("Mean F1: ", np.mean(f1s)*100)
print("F1 SD: ", np.std(f1s)*100)