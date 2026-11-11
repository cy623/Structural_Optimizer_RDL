Structural Optimizer RDL
=======

This code loads our trained weights and reproduces their predictions on RelBench classification tasks. The code of the paper "What Makes a Desired Graph for Relational Deep Learning?"


Setup
-------
Python 3.10 to 3.12 with CUDA 12.6. Run `sh install.sh` (a `.venv`) or `conda env create -f environment.yml`. Main versions: torch 2.7.1, torch-geometric 2.6.1, pytorch-frame 0.2.5, pyg-lib 0.4.0, torch-scatter 2.1.2.

The first run of each command downloads the data it needs into `data/raw`. `--mode prepare` builds the graph of a task in `data/graph`.

Predict with the released weights
-------

Put each task's `best.pt` in `<weights>/best_model/<dataset>__<task>/all/`, then:
```
python main.py --mode prepare --dataset rel-stack --task user-engagement
python main.py --mode predict --dataset rel-stack --task user-engagement --output_root <weights>
```
The predictions are written to `<weights>/predictions/<dataset>__<task>.csv`. The CSV has the task's key columns and one probability column.

Train
-------
```
python main.py --mode prepare --dataset rel-f1 --task driver-top3
python main.py --mode train --dataset rel-f1 --task driver-top3 --config config.json
```

