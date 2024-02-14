# Evaluate counterfactual explanibility methods, an attempt to extend to R-GCN

This repository follow the code of the official implementation of the [AISTATS 2022 paper CF-GNNExplainer: Counterfactual Explanations for Graph Neural Networks](https://arxiv.org/abs/2102.03322). 

# Installation

## From Source

Start by grabbing this source code:

```
git clone 
```

It is recommended to run this code inside a `virtualenv` with `python3.10`.

```
virtualenv venv -p /usr/local/bin/python3
source venv/bin/activate
```

### Requirements

To install all the requirements, run the following command:

```
python -m pip install -r requirements.txt
```

## Training original models

We can train the original models from model/Model.py. There are different possibilities so make sure to set either the tree dataset with the GCNsynthetic, the citeseer dataset with the GCN3Layer, or the AIFB dataset with the R-GCN. cd into model and run this command:

```train
python model.py
```

## Training and evaluate CF-GNNExplainer
Run the following commands from the original path of the project :

```train
python main_synthetic.py 
python main_cora.py 
python main_citeseer.py 
python main_RGCN.py 
```

## Results

Our model achieves the following performance for the tree dataset:

| Model name         | Dataset        | Fidelity       |  Size |    Sparsity   | Accuracy    |
| ------------------ |---------------- | -------------- | -------------- | -------------- |   -------------- |
| CF-GNNExplainer   |     Tree-Cycles  |      0.206       |      2.16           |       0.894        |      0.956       |
| CF-GNNExplainer   |     Citeseer  |      0.206       |      2.16           |       0.894        |      0.956       |



