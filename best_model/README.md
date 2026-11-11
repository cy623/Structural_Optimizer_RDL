Download the weights from the [model_weight release](https://github.com/cy623/Structural_Optimizer_RDL/releases/tag/model_weight) and unzip them into `outputs/`, where `main.py` looks for them:
```
wget https://github.com/cy623/Structural_Optimizer_RDL/releases/download/model_weight/structural_optimizer_rdl_weights.zip
unzip structural_optimizer_rdl_weights.zip -d outputs
python main.py --mode prepare --dataset rel-stack --task user-engagement
python main.py --mode predict --dataset rel-stack --task user-engagement
```
