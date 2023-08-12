import torch
import torch.nn as nn, torch.nn.functional as F
from torch.nn import Linear
from time import time


class MLP(nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels):
		super(MLP, self).__init__()
		self.linears = nn.ModuleList()

		if hidden_channels is not None:
			self.linears.append(Linear(in_channels, hidden_channels))
			self.linears.append(Linear(hidden_channels, out_channels))
		else:
			self.linears.append(Linear(in_channels, out_channels))

	def forward(self, x):
		for layer in self.linears[:-1]:
			x = F.relu(layer(x))
		x = self.linears[-1](x)
		return x



class kPathGNN(nn.Module):
	def __init__(self, k, in_channels, hidden_channels, out_channels, device='cpu') -> None:
		super(kPathGNN, self).__init__()

		# Save parameters
		self.k = k
		self.in_channels = in_channels
		self.hidden_channels = hidden_channels
		self.out_channels = out_channels
		self.device = device

		# Define layers
		self.self_layer = MLP(in_channels, hidden_channels, out_channels).to(self.device)
		self.layers = nn.ModuleList()
		for _ in range(k):
			self.layers.append(MLP(in_channels, hidden_channels, out_channels)).to(self.device)

		self.softmax = nn.Softmax(dim=1)
	
	def gather(self, x, edge_index):
		# Gather the k-hop neighbors of each node, without repetitions
		# x: [num_nodes, in_channels]
		# edge_index: [2, num_edges]

		# Initialize the result
		result = torch.zeros((x.shape[0], self.k, self.in_channels)).to(self.device)  # [num_nodes, k, in_channels]
		
		prev_adj = torch.eye(x.shape[0]).to(self.device)  # [num_nodes, num_nodes]
		adj = torch.zeros((x.shape[0], x.shape[0])).to(self.device)  # [num_nodes, num_nodes]
		adj[edge_index[0], edge_index[1]] = 1
		adj += prev_adj
		curr_adj = adj

		# Gather the k-hop neighbors
		for i in range(self.k):
			khops = ((curr_adj > 0)*(prev_adj == 0)).float()
			result[:, i, :] = torch.matmul(khops, x)
			prev_adj = curr_adj
			curr_adj = torch.matmul(curr_adj, adj)
		
		return result


	def forward(self, graph, valid = False):

		x = graph.x
		edge_index = graph.edge_index
		gathered = self.gather(x, edge_index)
		
		result = torch.zeros((x.shape[0], self.out_channels)).to(self.device)  # [num_nodes, out_channels]

		result += self.self_layer(x)
		for i in range(len(self.layers)):
			result += self.layers[i](gathered[:, i, :].squeeze())
		
		result = self.softmax(result)
		if valid:
			result = torch.argmax(result, dim=1)
		
		return result