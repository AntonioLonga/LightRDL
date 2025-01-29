# Boosting Relational Deep Learning with Pretrained Tabular Models
## Official Source Code for ICML 2025

This repository is organized into the following sections:

1. Environment Setup
2. Graph Construction
3. Model Training
3. Folder structure

To run the code, follow these steps:

1. Install the environment (Section 1)
2. Generate the static network (Section 2)
3. Train the model (Section 3)

---

# 1. Environment Installation
Bellow you can find the versions of the used pakages:

- pytorch=2.2.2
- pyg-lib==0.4.0+pt22cu121
- pytorch-frame==0.2.3
- relbench==1.1.0

---

# 2. Graph Construction
To build the graphs for a specific dataset and task, use the following command:

```bash
python graph_construction.py --dataset=rel-f1 --task=driver-dnf
```
The generated graphs are saved in the following directory structure:

```bash
static_networks/dataset_name/task_name/data_obj/file_name.pth
```
Each file name indicates the position of the static graph and whether it's for training, testing, or validation.


---

# 3. Model Training

To train the model on the **rel-f1** dataset for the **driver-dnf** task, use the following command:

```bash
python train_model.py
```

Note that **train_model.py** accepts the following parameters:

- **--dataset**: The name of the dataset. The default is `rel-f1`.
- **--task**: The name of the task. The default is `driver-dnf`.
- **--batch_size**: The batch size. The default is `1`.
- **--num_layers**: The number of layers in the model. The default is `3`.
- **--dropout_prob**: The dropout probability. The default is `0.1`.
- **--lr**: The learning rate. The default is `0.0001`.
- **--hidden_channels**: The dimension of the hidden channels. The default is `64`.
- **--weight_decay**: The weight decay rate. The default is `0.000001`.
- **--device**: The device to use (e.g., `cuda`). The default is `cuda`.
- **--compute_val_every**: Specifies how often to compute validation (in epochs). The default is `10`.
- **--patience**: Specifies the patience for early stopping. The default is `50`.
- **--nb_epochs**: The number of epochs to train. The default is `1000`.
- **--mode**: Specifies whether the model is in `training` or `testing` mode. The default is `training`.
- **--task_type**: Defines whether the task is `CLASSIFICATION` or `REGRESSION`. The default is `CLASSIFICATION`.
- **--target_table**: The name of the table containing the target, as specified in the rel-bench repository. The default is `"drivers"`.


by default the **train_model.py** script would load the static graphs computed in section (2) stored in:
```
static_networks/dataset_name/task_name/
```


# 3. Folder structure

The folders in this repository are organized as follows:

```bash
trained_models/model_name.pth
```
This folder contains the best model based on validation performance.



```bash
static_networks/dataset_name/task_name
```
This folder stores distilled embeddings and the PyG HeteroData object.


```bash
static_networks/dataset_name/embeddings
```
This folder holds the node feature embeddings. Categorical features are one-hot encoded, while numerical features are scaled using sklearn's StandardScaler.