# Federated Deep Regression Forests: Federated Learning Framework integrating REFINED CNN and Deep Regression Forests
Federated Deep Regression Forests are an adaption of Deep Neural Decision Forests [[1](https://doi.org/10.1109/ICCV.2015.172)] for regression [[2](https://doi.org/10.1109/CVPR.2018.00245)] in the federated setting. This repository contains a PyTorch implementation of Federated Deep Regression Forests and scripts to run a federated simulation. The simulations can be used for training and evaluating different models on the Cancer Cell Line Encyclopedia (CCLE) [[3](https://depmap.org/portal/download/)] dataset with varying degrees of heterogeneity among the clients. The initial deep regression code was acquired from [[4](https://github.com/Nicholasli1995/VisualizingNDF)] and extended to a federated framework simulation with additional competing models.

## Dependencies
Python packages required to run the simulation:
* PyTorch (https://pytorch.org/)
* pandas (https://pandas.pydata.org/docs/getting_started/install.html)
* NumPy (https://numpy.org/install/)
* scikit-learn (https://scikit-learn.org/stable/install.html)

The packages can be downloaded using pip or conda

The code has been tested with the following environment and versions:
* Windows 10
* NVIDIA 3080 Ti
* Cuda 11.6
* Python 3.8.5
* PyTorch 1.13.1
* pandas 1.5.3
* NumPy 1.21.0
* scikit-learn  1.2.1

## Running Instructions
After cloning this repository, use the following commands within the CCLE_NonIID_Python directory.

To run federated training:
```
python runSim.py -train True -model_type 'FedDRF' -iid 8
```
To evaluate the federated model:
```
python runSim.py -train False -eval True -model_type 'FedDRF' -iid 8
```
To train a centralized version of the model:
```
python runSim.py -train False -train_central True -model_type 'FedDRF' -iid 8
```
To evaluate the centralized version of the model:
```
python runSim.py -train False -eval_central True -model_type 'FedDRF' -iid 8
```
## Parameters (Defaults)
```
-train (True): Boolean to perform federated training or not
-train_central (False): Boolean to perform centralized training or not
-eval (False): Boolean to perform federated evaluation or not
-eval_central (False): Boolean to perform centralized evaluation or not
-model_type ('FedDRF'): Model to train, ('ANN', 'CNN', 'FedDRF')
-iid (2): Parameter to control heterogeneity of federated dataset, can be varied from 2-24 
```
Parameters train, train_central, eval, and eval_central are all Booleans that control the behavior of the simulation (i.e. whether to perform training or evaluation in the centralized or federated setting). The model_type parameter selects the model to train or evaluate and the iid parameter controls the heterogeneity of samples across clients with 2 being fully disjoint and 24 being a completely equal distribution. 

## Optional Parameters (Defaults)
Paths
```
-data_path ('../data/CCLE'): Path to CCLE data
-save_dir ('trainedModels_'): Directory to save trained models and validation results
-save_name ('trained_model'): Name of saved files
```
Training & Model Settings 
```
-batch_size (32): Batch size for training
-eval_batch_size (1000): Evaluation batch size
-leaf_batch_size (1000): Batch size for leaf node updates
-cuda (0): Boolean to use gpu for training or not
-gpuid (0): ID of the GPU to use for training
-epochs (200): Max number of epochs
-num_threads (0): Number of threads to use when loading data
-randomSeed (2022): Random seed for data partition
```
FedDRF Model Parameters 
```
-num_output (150): Neural feature output size for forest 
-n_tree (10): Number of trees in the forest
-tree_depth (7): Depth of trees
-label_iter_time (10): Number of iterations for updating leaf node parameters
```
Optimizer Settings
```
-lr (.001): Learning rate
-EStol (.0001): Early stopping tolerance
-ESpat (8): Early stopping patience
-LRsched (4): Learning rate scheduler patience
```
## Output Files
When running the script to train the models, the models and their validation progress will be saved into a folder in the working directory. All other outputs are printed to the console.

## Citation
If you find this package useful and end up using it, please cite the following work:
1) Nolte, D., Bazgir, O., Ghosh, S., & Pal, R. (2023). Federated learning framework integrating REFINED CNN and Deep Regression Forests. Bioinformatics advances, 3(1), vbad036. https://doi.org/10.1093/bioadv/vbad036
