import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GCN

class GCNModel(nn.Module):
	def __init__(self, n_layers, input_size, hidden_size, device) -> None:
		super().__init__()

		self.device = device
		self.convs = GCN(input_size, hidden_size, n_layers, 2, norm = 'batch').to(device)

		self.softmax = nn.Softmax(dim=1)
		self.relu = nn.ReLU()

	def forward(self, graph, valid = False):
		
		x = graph.x
		
		x = self.convs(x, graph.edge_index)
		x = self.softmax(x)

		if valid:
			x = torch.argmax(x, dim=1)
		return x