Overview
-------
This is the official repository of paper "What Makes a Desired Graph for Relational Deep Learning?"

Setup
-------
* torch==2.7.1
* torch-geometric==2.6.1
* relbench==1.1.0
* torch_scatter==2.1.2
* torch_frame==0.2.5
* numpy==2.1.2

Data Preparation
-------
You can install RelBench using：
```
pip install relbench
```
Get datasets：
```
python dataset.py
```
For the full details about RelBench: [RelBench](https://github.com/snap-stanford/relbench/tree/main)

Run pipeline
--------
1. You can run our model by the script:
```
python main.py
```
