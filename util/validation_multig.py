import torch

def validate_model(model, dataset, train_split, valid_split, debug=False, db=False):
	model.eval()
	pred = []
	for i in range(train_split, valid_split):
		graph = dataset.get(i).to(model.device)
		if not db:
			pred.append(torch.argmax(model(graph), dim=1)==graph.y)
		else:
			pred.append(torch.argmax(model(graph)[0], dim=1)==graph.y)
	pred = torch.cat(pred, dim=0).float()
	
	perc = pred.mean().item()*100

	mess = f"{perc:.2f}%"
	return perc, mess


def test_model(model, dataset, valid_split, test_split, debug=False, db=False):
	model.eval()
	pred = []
	true_positive = 0
	true_negative = 0
	false_positive = 0
	false_negative = 0

	for i in range(valid_split, test_split):
		graph = dataset.get(i).to(model.device)
		prediction = model(graph)[0] if db else model(graph)
		prediction = torch.argmax(prediction, dim=1)

		true_positive += torch.sum(prediction[graph.y==1]==1).item()
		true_negative += torch.sum(prediction[graph.y==0]==0).item()
		false_positive += torch.sum(prediction[graph.y==0]==1).item()
		false_negative += torch.sum(prediction[graph.y==1]==0).item()

	n_all = true_positive + true_negative + false_positive + false_negative
	
	accuracy = (true_positive + true_negative)/n_all
	precision = true_positive/(true_positive + false_positive)
	recall = true_positive/(true_positive + false_negative)
	f1 = 2*precision*recall/(precision + recall)

	if debug:
		print(f"Test accuracy: {accuracy:.2f}%")
		print(f"Test precision: {precision:.2f}%")
		print(f"Test recall: {recall:.2f}%")
		print(f"Test F1: {f1:.2f}%")
		
	return accuracy, precision, recall, f1



def evaluate_model(model, dataset, train_split, valid_split, test_split, debug=False, db=False):
	val_acc = validate_model(model, dataset, train_split, valid_split, debug=False, db=db)
	acc, prec, rec, f1 = test_model(model, dataset, valid_split, test_split, debug=False, db=db)

	return val_acc, acc, prec, rec, f1