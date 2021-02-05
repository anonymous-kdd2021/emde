## EMDE

Install as a package:
```
pip install --editable ./
```

## Scripts descriptions
`coders/dlsh.py` - implementation of manifold partitioning using Density-dependent LSH (DLSH) algoritm

`coders/cleora.py` -  utils for simple network embedding method with the desired property of local similarity for both item attributes and interaction data.

`datasets/sessionBasedDataset.py` - pyTorch dataset for session-based datasets. It return two input sketches to the network: one represents all items that user has interaction with (except the last one), the second sketch represents only last single item that user has interaction with.

`loaders/sessionBasedLoaders.py` - loading data from session-based datasets (such as retail, digi, rsc15) and converts them to the format accepted by `sessionBasedDataset.py`  

`models/resnetNetwork.py` - 3 layer feed forward netowrk with residual connections and batch normalization. Used in session based recommendation model.

# TO DO
* add simple example for some dataset with our evaluation metrics
* add cleora-light binary file or use cleora in another way
* dockerize
* think about moving loaders and datasets to session-rec repo
