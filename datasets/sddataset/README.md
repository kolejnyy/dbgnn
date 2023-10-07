# Signal Distribution Dataset (SDDataset)

## Dataset specification

The dataset consists of artificially generated graphs, where nodes represent distribution points and edges represent connections. Each node has a set of features, being a 1-hot encoding of the strength of the source of signal / distractor at that point. We assume that source of signal $v$ distributes signal to all points located in a radius of $k_v$, where $k_v$ is the strength of source $v$, with the strenth decreasing by 1 as the distance from the source increases by 1. Analogously, the distractors affect points lying in at most $|k_v|$ distance from the distractor, where $k_v$ is the strength of the distractor. The strength of the distractor is negative, and the strength of the signal is positive. 

The targets are binary labels, where a label of 1 means that there is a signal at the point (the cumulative power of signal sources at that exceeds the strength of distraction at that point).

For each such dataset, we generate $1000$ graphs. The graphs are taken from the PATTERN dataset, and the features are generated randomly, according to the pre-defined probability distribution. The labels are then generated according to the specification.

The specification of the saved k5_g1000 dataset is:

```
k_max 	= 5
rho		= np.array([1,2,4,8,16,128,16,8,4,2,1])
```


## Results

The results of the experiments are presented in the table below. GCN models consisted of 5 layers with hidden dimension of 32 and were trained for 100 epochs with learning rate 0.003 and batch size 32. For each evaluation the model maximizing validation accuracy was chosen for predicting the test split. The results shown below are taken over 50 evaluations.

| Model | Accuracy | Precision | Recall | F1 |
| --- | :---: | :---: | :---: | :-----: |
| GCN		|	91.24 ± 0.53	|	90.66 ± 1.17	|	89.22 ± 1.66	|	89.92 ± 0.67	|
| DB-GCN	|	89.91 ± 0.29	|	89.80 ± 1.08	|	86.86 ± 1.52	|	88.29 ± 0.42	|
| DB-GCN+SV |   89.66 ± 0.28 	|	90.55 ± 0.89	|	85.34 ± 1.49	|	87.85 ± 0.46	|
