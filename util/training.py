import numpy as np
import copy

import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt

from util.validation_singleg import validate_model as validate_single, evaluate_model as evaluate_single
from util.validation_multig import validate_model as validate_multi, evaluate_model as evaluate_multi
from util.thresholding import generate_threshold_map

def create_mini_batch(graph_list):
	batch_edge_index = graph_list[0].edge_index
	batch_x = graph_list[0].x
	batch_y = graph_list[0].y
	batch_thr = graph_list[0].thr
	batch_batch = torch.zeros((graph_list[0].num_nodes), dtype=torch.int64)

	nodes_count = graph_list[0].num_nodes

	for idx, graph in enumerate(graph_list[1:]):
		batch_x = torch.cat((batch_x, graph.x))
		batch_y = torch.cat((batch_y, graph.y))
		batch_thr = torch.cat((batch_thr, graph.thr))

		batch_edge_index = torch.cat((
			batch_edge_index, 
			torch.add(graph.edge_index, nodes_count))
		, dim=1)
		nodes_count += graph.num_nodes

		batch_batch = torch.cat((
			batch_batch, 
			torch.full((graph.num_nodes,), idx + 1)
		))

	batch_graph = Data(x=batch_x, edge_index=batch_edge_index, y=batch_y, batch=batch_batch, thr=batch_thr.reshape(-1, 1))
	return batch_graph




def train(model, dataset, epochs, batch_size, lr, train_split, valid_split, test_split,
		weight_decay=0, print_interval=10, valid_interval=10, debug=False,
		db=False, thr_sv=False, thr_sv_alpha=10, single_graph=False):

	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	criterion = torch.nn.CrossEntropyLoss()
	mse = torch.nn.MSELoss()

	best_model = None
	best_val_acc = 0

	total_losses = []
	val_accuracies = []

	graph = dataset.get(0)
	graph.thr = generate_threshold_map(graph)
	graph = graph.to(model.device)

	for epoch in range(epochs):
		model.train()
		losses = []
		if not single_graph:
			perm = torch.randperm(train_split)
			for i in range(0, train_split, batch_size):
				optimizer.zero_grad()
				graph_list = [dataset.get(perm[i+j]) for j in range(min(batch_size, train_split-i))]
				batch = create_mini_batch(graph_list).to(model.device)
				if not db:
					out = model(batch)
				else:
					out, pred, thr = model(batch)

				loss = criterion(out, batch.y)
				if db:
					loss += criterion(pred, batch.y)
				if thr_sv:
					loss += thr_sv_alpha*mse(thr, batch.thr)

				loss.backward()
				optimizer.step()
				losses.append(loss.item())
		else:
			optimizer.zero_grad()
			if not db:
				out = model(graph)
			else:
				out, pred, thr = model(graph)
			
			loss = criterion(out[train_split], graph.y[train_split])
			if db:
				loss += criterion(pred[train_split], graph.y[train_split])
			if thr_sv:
				loss += thr_sv_alpha*mse(thr[train_split].flatten(), graph.thr[train_split])

			loss.backward()
			optimizer.step()
			losses.append(loss.item())


		total_losses.append(np.mean(losses))

		if (epoch+1)%valid_interval==0:
			val_acc, mess = validate_single(model, dataset, valid_split, db=db) if single_graph else \
							validate_multi(model, dataset, train_split, valid_split, db=db)
			val_accuracies.append(val_acc)
			if val_acc > best_val_acc:
				best_val_acc = val_acc
				best_model = copy.deepcopy(model)

		if (epoch+1)%print_interval==0 and debug:
			message = f"Epoch {epoch+1}/{epochs}:   loss: {np.mean(losses):.4f}"
			message += " - val_acc: " + str(mess)
			print(message)

	if debug:
		plt.plot(total_losses, label="Training loss")
		plt.show()

		plt.plot(list(range(0, epochs, valid_interval)), val_accuracies, label="Validation accuracy")
		plt.show()

	val_acc, acc, prec, rec, f1 = 	evaluate_single(best_model, dataset, valid_split, test_split, db=db) if single_graph \
									else evaluate_multi(best_model, dataset, train_split, valid_split, test_split, db=db)
	return best_model, val_acc, acc, prec, rec, f1