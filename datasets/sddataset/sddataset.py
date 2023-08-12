import numpy as np

import torch
from torch_geometric.datasets import FakeDataset
from torch_geometric.data import Dataset, Data

from tqdm import tqdm


class SDDataset(Dataset):
	def __init__(self, k_max=5, load_path=None, rho=None, transform=None, pre_transform=None,
				 num_graphs=1000, avg_num_nodes=100, avg_degree=6):
		super().__init__(None, transform, pre_transform)
		
		self.k_max = k_max
		
		self.rho = rho
		if self.rho is None:
			self.rho = np.ones(2*k_max+1)/(2*k_max+1)
		if self.rho.shape[0] != 2*k_max+1:
			raise ValueError("rho must have length k_max+1")
		self.rho = self.rho/np.sum(self.rho)
		self.data = []

		if load_path is not None:
			x = np.load(load_path+'x.npy', allow_pickle=True)
			y = np.load(load_path+'y.npy', allow_pickle=True)
			edge_index = np.load(load_path+'edge_index.npy', allow_pickle=True)
			for i in tqdm(range(len(x))):
				graph = Data(x=torch.from_numpy(x[i].reshape(-1, 2*k_max+1)).float(), y=torch.from_numpy(y[i]).float(), edge_index=torch.from_numpy(edge_index[i].reshape(2, -1)).long())
				self.data.append(graph)

		else:
			self.fake_data = FakeDataset(num_graphs = num_graphs, avg_num_nodes=avg_num_nodes, avg_degree=avg_degree, num_channels=k_max, num_classes = 2, task='node')
			for i in tqdm(range(len(self.fake_data))):
				graph = self.fake_data.get(i)
				graph = self.prepare(graph)
				self.data.append(graph)

	def prepare(self, graph):
		
		# Generate node features
		features = np.random.choice(list(range(-self.k_max, self.k_max+1)), size=graph.num_nodes, p=self.rho)
		onehot_features = np.zeros((graph.num_nodes, 2*self.k_max+1))
		onehot_features[np.arange(graph.num_nodes), features+self.k_max] = 1
		graph.x = torch.from_numpy(onehot_features).float()
		
		# Generate node labels
		graph.y, _ = propagate_signal(graph, features)
		return graph

	def len(self):
		return len(self.fake_data)
	
	def get(self, idx):
		return self.data[idx]



def propagate_signal(graph, signal):
	"""Propagate signal through the graph"""

	strength = torch.zeros(graph.num_nodes)
	used = torch.zeros(graph.num_nodes)

	neighbors = [[] for _ in range(graph.num_nodes)]
	for i in range(graph.num_edges):
		neighbors[graph.edge_index[0][i]].append(graph.edge_index[1][i])
		neighbors[graph.edge_index[1][i]].append(graph.edge_index[0][i])

	for i in range(graph.num_nodes):
		used = torch.zeros(graph.num_nodes)
		queue = [(i, signal[i])]
		while len(queue) > 0:
			node, sgn = queue.pop(0)
			if used[node] == 1 or sgn == 0:
				continue
			used[node] = 1
			strength[node] += sgn
			step = -1 if sgn > 0 else 1
			for neighbor in neighbors[node]:
				queue.append((neighbor, sgn+step))
	
	return (strength>0).long(), strength