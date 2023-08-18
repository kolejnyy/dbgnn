import numpy as np

import torch
from torch_geometric.datasets import FakeDataset
from torch_geometric.data import Dataset, Data

from tqdm import tqdm

import networkx as nx
from torch_geometric.utils import from_networkx, to_networkx
from torch import from_numpy

class IslandDataset(Dataset):
	def __init__(self, load_path=None,
				n_graphs=1000, grid_size=30, embed_size = 10,
				gen_steps_min=100, gen_steps_max=200,
				transform=None, pre_transform=None):
		super().__init__(None, transform, pre_transform)
		
		self.load_path = load_path
		self.data = []
		self.n_graphs = n_graphs
		self.grid_size = grid_size
		self.embed_size = embed_size

		self.gen_steps_min = gen_steps_min
		self.gen_steps_max = gen_steps_max

		if load_path is not None:
			x = np.load(load_path+'x.npy', allow_pickle=True)
			y = np.load(load_path+'y.npy', allow_pickle=True)
			thr = np.load(load_path+'thr.npy', allow_pickle=True)
			edge_index = np.load(load_path+'edge_index.npy', allow_pickle=True)
			for i in tqdm(range(len(x))):
				graph = Data(x=torch.from_numpy(x[i].reshape(-1, embed_size).astype(np.float32)).float(),
							y=torch.from_numpy(y[i]).long(),
							edge_index=torch.from_numpy(edge_index[i].reshape(2, -1)).long(),
							thr=torch.from_numpy(thr[i]).float())
				self.data.append(graph)
		
		else:
			self.data = [self.generate_graph() for _ in range(n_graphs)]
			
	def generate_threshold_map(self, graph):
		new_thr = torch.zeros(graph.x.shape[0])
		for idx in range(graph.x.shape[0]):
			neighbours = graph.edge_index[1][graph.edge_index[0]==idx]
			if torch.max(graph.y[neighbours]) == 1 and graph.y[idx] == 0:
				new_thr[idx] = 1
		for idx in range(graph.x.shape[0]):
			neighbours = graph.edge_index[1][graph.edge_index[0]==idx]
			if torch.max(new_thr[neighbours]) == 1 and new_thr[idx] == 0:
				new_thr[idx] = 0.5
		return new_thr 
	
	def generate_graph(self):
		res = np.zeros((self.grid_size,self.grid_size))-1
		i = np.random.randint(0,self.grid_size)
		j = np.random.randint(0,self.grid_size)

		plama_size = np.random.randint(self.gen_steps_min,self.gen_steps_max)
		for _ in range(plama_size):
			res[i,j] = 1
			i += np.random.randint(-1,2)
			j += np.random.randint(-1,2)
			i = np.clip(i,0,self.grid_size-1)
			j = np.clip(j,0,self.grid_size-1)

		for i in range(self.grid_size):
			for j in range(self.grid_size):
				if res[i,j]!=1 and (res[np.clip(i-1,0,self.grid_size-1),j] == 1 or res[np.clip(i+1,0,self.grid_size-1),j] == 1 or res[i,np.clip(j-1,0,self.grid_size-1)] == 1 or res[i,np.clip(j+1,0,self.grid_size-1)] == 1):
					res[i,j] = 0.5

		prob_map = np.zeros((self.grid_size,self.grid_size))
		prob_map[res>=0.5]=1


		x = np.random.rand(self.grid_size,self.grid_size,self.embed_size)
		x[prob_map==0,:]*=-1

		G = nx.grid_2d_graph(self.grid_size, self.grid_size)
		graph = from_networkx(G)
		graph.x = from_numpy(x.reshape(-1, self.embed_size)).float()
		graph.y = from_numpy(prob_map.reshape(-1)).long()
		graph.thr = self.generate_threshold_map(graph)

		return graph

	def len(self):
		return len(self.data)
	
	def get(self, idx):
		return self.data[idx]