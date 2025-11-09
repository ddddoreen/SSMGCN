# SSMGCN
Installation

We recommend using Python >= 3.8 with PyTorch >= 1.10. The commands are as follows:

conda create -n ssmgcn python=3.8
conda activate ssmgcn
pip install -r requirements.txt  # install dependencies

Datasets

This project uses several publicly available spatial transcriptomics datasets (e.g., DLPFC, BRCA, mouse brain, etc.).
Please organize the data under the data/ directory following the example structure in this repository.

For each dataset, you typically need:

Gene expression matrix (e.g., counts.csv / expression.csv)

Spatial coordinates (e.g., spatial.csv)

(Optional) Ground-truth spatial domain / region labels (e.g., labels.csv)

Example structure:

data/
├── DLPFC/
│   ├── 151507/
│   │   ├── counts.csv
│   │   ├── spatial.csv
│   │   └── labels.csv   # if ground-truth labels are available
│   └── ...
└── BRCA/
    └── ...


As long as you keep the same file names and directory layout, the provided config files can be used directly for training and evaluation.

Training

Most training configurations can be modified in the config files under the configs/ directory
(e.g., model settings, dataset name, batch size, learning rate, number of epochs, etc.).

For example, to train SSMGCN on a DLPFC slice:

python src/train_ssmgcn.py --config configs/dlpfc_151507.yaml


Key parameters include:

dataset / data_path: which dataset to use

n_clusters: number of clusters / spatial domains

epochs, batch_size, lr: training hyperparameters

device: running device (e.g., cuda:0 or cpu)

You can also use shell scripts under the scripts/ directory to run experiments on multiple datasets or slices.

Evaluation

After training, you can evaluate the clustering performance and generate visualization results.

Example command:

python src/eval_ssmgcn.py --config configs/dlpfc_151507.yaml --ckpt path/to/checkpoint.pth


Main parameters include:

ckpt: path to the saved model checkpoint

metrics: evaluation metrics (e.g., ARI, NMI, SC, DB, etc.)

save_dir: directory to save clustering labels, logs, and visualization figures

Some scripts may combine training and evaluation in a single run; please refer to the comments in the scripts/ directory for details.

Acknowledgement

This project is implemented based on:

PyTorch and the standard Python scientific computing ecosystem

Optionally, libraries such as PyTorch Geometric / Scanpy for graph construction and preprocessing

We also thank the authors and maintainers of the original spatial transcriptomics datasets for making their data publicly available.
