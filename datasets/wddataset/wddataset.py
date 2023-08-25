import numpy as np

import torch
from torch_geometric.datasets import FakeDataset
from torch_geometric.data import Dataset, Data

from tqdm import tqdm

from torch_geometric.utils import to_dense_adj


class WDDataset(Dataset):
	def __init__(self, k_max=7, load_path=None, rho=None, transform=None, pre_transform=None,
				 num_graphs=1000, avg_num_nodes=100, avg_degree=6):
		super().__init__(None, transform, pre_transform)
		
		self.k_max = k_max
		self.rho = rho
		if self.rho is None:
			self.rho = np.ones(k_max+1)/(k_max+1)
		if self.rho.shape[0] != k_max+1:
			raise ValueError("rho must have length k_max+1")
		self.data = []

		if load_path is not None:
			x = np.load(load_path+'x.npy', allow_pickle=True)
			y = np.load(load_path+'y.npy', allow_pickle=True)
			edge_index = np.load(load_path+'edge_index.npy', allow_pickle=True)
			for i in tqdm(range(len(x))):
				graph = Data(x=torch.from_numpy(x[i].reshape(-1, k_max)).float(), y=torch.from_numpy(y[i]).long(), edge_index=torch.from_numpy(edge_index[i].reshape(2, -1)).long())
				self.data.append(graph)

		else:
			self.fake_data = FakeDataset(num_graphs = num_graphs, avg_num_nodes=avg_num_nodes, avg_degree=avg_degree, num_channels=k_max+1, num_classes = 2, task='node')
			for i in range(len(self.fake_data)):
				graph = self.fake_data.get(i)
				graph = self.prepare(graph)
				self.data.append(graph)


	def prepare(self, graph):
		
		# Generate node features
		features = np.random.choice(self.k_max+1, size=graph.num_nodes, p=self.rho)
		onehot_features = np.zeros((graph.num_nodes, self.k_max+1))
		onehot_features[np.arange(graph.num_nodes), features] = 1
		graph.x = torch.from_numpy(onehot_features).float()
		
		# Generate node labels
		adj = to_dense_adj(graph.edge_index)[0]
		features = graph.x
		shifted = torch.zeros_like(features)
		shifted[:,:-1] = features[:,1:]
		for i in range(self.k_max):
			features = features + torch.mm(adj, shifted)
		graph.y = torch.sum(features[:,1:], dim=1).long()
		graph.y = (graph.y > 0).long()
		return graph

	def len(self):
		return len(self.fake_data)
	
	def get(self, idx):
		return self.data[idx]


