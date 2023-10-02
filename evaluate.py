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
from torch_geometric.datasets import GitHub

from util.visualisation import *
from util.training import train
from util.thresholding import generate_threshold_map


# =============================================================================
#								CONFIGURATION
# =============================================================================

# Dataset name, one of: "sddataset", "island", 'twitch', 'pubmed', 'github'
dataset_name = 'github'

# Model type, one of: 'gcn'
model_type = 'gcn'
n_layers = 5
hidden_size = 128
device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

# Differentiable binarisation module specification
db = False
thr_sv = False

# Number of evaluations to perform
evaluations = 10

# Training specification
epochs = 30
batch_size = 32
lr = 0.005
val_interval = 1


# =============================================================================
#								INITIALISATION
# =============================================================================

dataset	= None
model 	= None
input_size = None

train_split = None
valid_split = None
test_split  = None

single_graph = False


if dataset_name == 'sddataset':
	# dataset = SDDataset(load_path='datasets/sddataset/k5_g1000/', k_max=5, rho=np.array([1,2,4,8,16,128,16,8,4,2,1]), num_graphs=1000, avg_num_nodes=100, avg_degree=4)
	dataset = SDDataset(load_path='datasets/sddataset/k5_g1000/', k_max=5)
	train_split = 800; valid_split = 900; test_split  = 1000
	input_size = 11

elif dataset_name == 'island':
	dataset = IslandDataset(load_path='datasets/island/30x30_g1000/')
	train_split = 800; valid_split = 900; test_split  = 1000
	input_size = 10

elif dataset_name == 'twitch':
	dataset = Twitch(root='datasets/twitch/', name='DE', transform=None)
	graph = dataset[0]
	test_split = np.arange(len(graph.x) // 10)*10
	valid_split = np.arange((len(graph.x)-1)//10)*10+1
	train_split = np.ones(len(graph.x), dtype=bool)
	train_split[test_split] = False
	train_split[valid_split] = False
	input_size = 128

elif dataset_name == 'pubmed':
	dataset = Planetoid(root='datasets/', name='Pubmed', transform=None)
	graph = dataset[0]
	test_split = np.arange(len(graph.x) // 10)*10
	valid_split = np.arange((len(graph.x)-1)//10)*10+1
	train_split = np.ones(len(graph.x), dtype=bool)
	train_split[test_split] = False
	train_split[valid_split] = False
	y = graph.y
	dataset[0].y[y == 2] = 1
	dataset[0].y = 1-dataset[0].y
	input_size = 500
	single_graph = True

elif dataset_name == 'github':
	dataset = GitHub(root='datasets/github/', transform=None)
	graph = dataset[0]
	test_split = np.arange(len(graph.x) // 10)*10
	valid_split = np.arange((len(graph.x)-1)//10)*10+1
	train_split = np.ones(len(graph.x), dtype=bool)
	train_split[test_split] = False
	train_split[valid_split] = False
	y = graph.y
	dataset[0].y[y == 2] = 1
	dataset[0].y = 1-dataset[0].y
	input_size = 128
	single_graph = True



print("Using device: ", device)

if model_type == 'gcn':
	if not db:
		model = GCNModel(n_layers, input_size, hidden_size, device)
	else:
		model = DBGNNModel(GCN(input_size, hidden_size, n_layers-2, hidden_size, norm = 'batch'),
								GCN(hidden_size, hidden_size, 2, 1, norm = 'batch'),
								GCN(hidden_size, hidden_size, 2, 1, norm = 'batch'),
								device=device)


# =============================================================================
#								EVALUATION
# =============================================================================

accuracies = np.zeros(evaluations)
precisions = np.zeros(evaluations)
recalls = np.zeros(evaluations)
f1s = np.zeros(evaluations)

for i in tqdm(range(evaluations)):
	_, _, acc, prec, rec, f1 = train(model, dataset, epochs, batch_size, lr,
				  					train_split, valid_split, test_split,
									print_interval=10, valid_interval=val_interval,
				  					db=db, thr_sv=thr_sv,
									single_graph=single_graph)
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