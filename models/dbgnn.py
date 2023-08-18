import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm


class DBGNNModel(nn.Module):
    
	def __init__(self, backbone_model, pred_model, thr_model, k=30, device='cpu') -> None:
		super().__init__()

		self.device = device
		self.backbone = backbone_model.to(self.device)
		self.pred = pred_model.to(self.device)
		self.thr = thr_model.to(self.device)
		
		self.sigmoid = nn.Sigmoid()
		self.softmax = nn.Softmax(dim=1)
		self.relu = nn.ReLU()

		self.k = k


	def forward(self, graph):
		
		# Calculate the probability of each node being a tap
		feats = self.backbone(graph.x, graph.edge_index)
		p = self.sigmoid(self.pred(feats, graph.edge_index))
		t = self.sigmoid(self.thr(feats, graph.edge_index))

		# Calculate the thresholded probability
		x = 1 / (1 + torch.exp(-self.k * (p - t)))

		# Calculate the final output
		x = torch.stack([1-x, x], dim=1).squeeze()
		return x, torch.stack([1-p,p], dim=1).squeeze(), t