import torch

def validate_model(model, dataset, valid_mask, debug=False, db=False):
	model.eval()
	pred = []
	graph = dataset.get(0).to(model.device)
	if not db:
		pred.append(torch.argmax(model(graph), dim=1)[valid_mask]==graph.y[valid_mask])
	else:
		pred.append(torch.argmax(model(graph)[0], dim=1)[valid_mask]==graph.y[valid_mask])
	
	pred = torch.cat(pred, dim=0).float()
	perc = pred.mean().item()*100

	mess = f"{perc:.2f}%"
	return perc, mess



def test_model(model, dataset, test_mask, debug=False, db=False):
	model.eval()
	true_positive = 0
	true_negative = 0
	false_positive = 0
	false_negative = 0

	graph = dataset.get(0).to(model.device)
	prediction = model(graph)[0][test_mask] if db else model(graph)[test_mask]
	prediction = torch.argmax(prediction, dim=1)
	labels = graph.y[test_mask]

	true_positive += torch.sum(prediction[labels==1]==1).item()
	true_negative += torch.sum(prediction[labels==0]==0).item()
	false_positive += torch.sum(prediction[labels==0]==1).item()
	false_negative += torch.sum(prediction[labels==1]==0).item()

	n_all = true_positive + true_negative + false_positive + false_negative
	
	accuracy = (true_positive + true_negative)/n_all
	precision = true_positive/max(true_positive + false_positive, 1)
	recall = true_positive/max(1, true_positive + false_negative)
	f1 = 2*precision*recall/max(0.0001, precision + recall)

	if debug:
		print(f"Test accuracy: {accuracy:.2f}%")
		print(f"Test precision: {precision:.2f}%")
		print(f"Test recall: {recall:.2f}%")
		print(f"Test F1: {f1:.2f}%")
		
	return accuracy, precision, recall, f1



def evaluate_model(model, dataset, valid_mask, test_mask, debug=False, db=False):
	val_acc = validate_model(model, dataset, valid_mask, debug=debug, db=db)
	acc, prec, rec, f1 = test_model(model, dataset, test_mask, debug=debug, db=db)

	return val_acc, acc, prec, rec, f1