# SSMGCN
## Installation

Python>=3.8.0 environment, including PyTorch>=1.8.
The command is as follows.

```bash
conda create -n SSMGCN python=3.8
conda activate SSMGCN
pip install -r requirements.txt  # install
```
## Datasets

This project uses several publicly available spatial transcriptomics datasets (e.g., DLPFC, BRCA, mouse brain, etc.).
Please place the data under the data/ directory.

## Training

Most training configurations can be changed in the config files under the configs/ directory.
The key factors include dataset, number of clusters, epochs, batch size, learning rate and device.


## Evaluation

After training, you can evaluate the clustering performance and generate visualization results.
The key factors include the model checkpoint, dataset, metrics and output path.


## Acknowledgement
Our code is implemented with PyTorch and common Python scientific libraries.
We also thank the authors of the original spatial transcriptomics datasets for making their data publicly available.
