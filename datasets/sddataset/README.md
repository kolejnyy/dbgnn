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