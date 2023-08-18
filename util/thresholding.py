import torch

def generate_threshold_map(graph):
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