# Differentiable Binarisation meets Graph Neural Networks

Introduced by Liao et. al in the architecture of [DBNet](https://arxiv.org/abs/1911.08947) and then extended by the same authors in [DBNet++](https://arxiv.org/pdf/2202.10304.pdf), Differentiable Binarisation (DB) proved itself to be a powerful and time efficient tool for binarising the output of a neural network. Originally applied to the task of "in-the-wild" text detection, where the binarisation of the output is a crucial and non-trivial step, the technique provided a way for neural networks to learn the binarisation process by themselves, without the need of manually-set post-processing steps. It increased the accuracy of classification of the pixels around the border of text instances, leading not only to a better detection, but also to a better segmentation.

Given the main goal of introducing DB was to support classification of the pixels close to the borders of text instances, it is natural to think that this technique could be applied to other tasks where the point belonging to one class lie in compact clusters, and the points on the boundaries can lead to misclassification. This potentially can be the case in the task of node classification in graph neural networks (GNNs), where the nodes belonging to one class are often grouped together in the graph. This repository implements the DB module in the context of GNNs, and checks whether it can improve the performance of several GNN architectures across multpile binary node classification datasets.

## Differentiable Binarisation

In the paper [DBNet](https://arxiv.org/abs/1911.08947), the authors propose a differentiable binarisation module that can be inserted in any neural network trained for binary classification problem. The original architecture of DBNet consists of three main components:
- backbone network (e.g. ResNet-50) used to extract features from the input image
- probability head, producing a segmentation map
- thresholding head, predicting the boundaries of text regions, i.e. the areas whose pixels should be strongly binarised

<img src="images/db_module.png" width=100%/>

To obtain the final binarised output, each pixel is transformed according to the DB formula:

$\hat{B}_{i,j} = \frac{1}{1+e^{-k(P_{i,j}-T_{i,j})}}$

where $P$ represents the segmetation map, $T$ - the thresholding map, and $k$ is a temperature constant, set by the authors to 50. This allows the network to learn the binarisation process on its own, without the need of manually setting the thresholding value, making the whole process differentiable and end-to-end trainable.

In the experiments, the same architecture is adapted to the GNN scenario: for a given backbone network and classification head, we study the effect of adding the DB module to the network.

## Datasets

The experiments are performed on the following real datasets:
- [Pubmed](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.CitationFull.html#torch_geometric.datasets.CitationFull)
- [Github](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.GitHub.html#torch_geometric.datasets.GitHub)
- [EllipticBitcoinDataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.EllipticBitcoinDataset.html#torch_geometric.datasets.EllipticBitcoinDataset)
- [Twitch-DE](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Twitch.html#torch_geometric.datasets.Twitch)

and two artificial datasets:

#### Water Distribution Dataset (WDD)

The dataset consists of artificially generated graphs, where nodes represent distribution points and edges represent pipes. Each node has a set of features, being a 1-hot encoding of the strength of the source of water at that point. We assume that source of water $v$ distributes water to all points located in a radius of $k_v$, where $k_v$ is the strength of source $v$.

The targets are binary labels, where a label of 1 means that the node is a tap, and a label of 0 means that the node is not a tap (no water is distributed to that point).

#### Signal Distribution Dataset (SDD)

The dataset consists of artificially generated graphs, where nodes represent distribution points and edges represent connections. Each node has a set of features, being a 1-hot encoding of the strength of the source of signal / distractor at that point. We assume that source of signal $v$ distributes signal to all points located in a radius of $k_v$, where $k_v$ is the strength of source $v$, with the strenth decreasing by 1 as the distance from the source increases by 1. Analogously, the distractors affect points lying in at most $|k_v|$ distance from the distractor, where $k_v$ is the strength of the distractor.

The targets are binary labels, where a label of 1 means that there is a signal at the point (the cumulative power of signal sources at that exceeds the strength of distraction at that point).


## Results

### Signal Distribution Dataset

5 layers, 100 epochs, 0.003 lr, 32 hidden dim, 32 batch_size, 50 evaluations

| Model | Accuracy | Precision | Recall | F1 |
| --- | :---: | :---: | :---: | :-----: |
| GCN		|	91.24 ± 0.53	|	90.66 ± 1.17	|	89.22 ± 1.66	|	89.92 ± 0.67	|
| DB-GCN	|	89.91 ± 0.29	|	89.80 ± 1.08	|	86.86 ± 1.52	|	88.29 ± 0.42	|
| DB-GCN+SV |   89.66 ± 0.28 	|	90.55 ± 0.89	|	85.34 ± 1.49	|	87.85 ± 0.46	|

### Island Dataset

5 layers, 100 epochs, 0.003 lr, 32 hidden dim, 12 batch_size, 25 evaluations 

| Model | Accuracy | Precision | Recall | F1 |
| --- | :---: | :---: | :---: | :-----: |
| GCN		|	99.77 ± 0.14	|	98.99 ± 0.70	|	99.33 ± 0.39	|	99.16 ± 0.51	|
| DB-GCN	|	99.89 ± 0.09	|	99.59 ± 0.54	|	99.63 ± 0.29	|	99.61 ± 0.31	|
| DB-GCN+SV |   99.80 ± 0.09 	|	99.08 ± 0.62	|	99.50 ± 0.17	|	99.29 ± 0.32	|

### Twitch-DE

5 layers, 200 epochs, 0.007 lr, 128 hidden dim, 50 evaluations

| Model | Accuracy | Precision | Recall | F1 |
| --- | :---: | :---: | :---: | :-----: |
| GCN		|	64.40 ± 0.97	|	68.54 ± 1.50	|	77.16 ± 5.45	|	72.45 ± 1.76	|
| DB-GCN	|	65.11 ± 0.84	|	68.99 ± 1.21	|	77.80 ± 4.50	|	73.04 ± 1.48	|
| DB-GCN+SV |   64.20 ± 1.55 	|	67.85 ± 2.29	|	79.24 ± 8.28	|	72.78 ± 2.72	|

### Pubmed

5 layers, 200 epochs, 0.005 lr, 256 hidden dim, 50 evaluations

| Model | Accuracy | Precision | Recall | F1 |
| --- | :---: | :---: | :---: | :-----: |
| GCN		|	92.91 ± 0.27	|	95.12 ± 0.30	|	95.85 ± 0.54	| 	95.48 ± 0.18	|
| DB-GCN	|	92.36 ± 0.21	|	94.94 ± 0.36	|	95.32 ± 0.42	|	95.13 ± 0.13	|
| DB-GCN+SV |	91.43 ± 0.48	|	94.49 ± 0.49	|	94.57 ± 0.75	|	94.53 ± 0.32	|

### GitHub

5 layers, 30 epochs, 0.005 lr, 128 hidden dim, 10 evaluations

| Model | Accuracy | Precision | Recall | F1 |
| --- | :---: | :---: | :---: | :-----: |
| GCN		|	85.82 ± 0.38	|	78.54 ± 2.34	|	63.88 ± 2.55	| 	70.38 ± 0.83	|
| DB-GCN	|	 ± 	|	 ± 	|	 ± 	|	 ± 	|
| DB-GCN+SV |	 ± 	|	 ± 	|	 ± 	|	 ± 	|

## TODO
- [ ] Add DB to other GNN architectures (GAT, GIN)
- [ ] Update READMEs with the results of the new experiments