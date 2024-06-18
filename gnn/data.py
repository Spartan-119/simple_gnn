import torch
from torch_geometric.datasets import Planetoid

def load_data(dataset_name = 'Cora', data_dir = '/tmp/Planetoid'):
    dataset = Planetoid(root = data_dir, name = dataset_name)
    data = dataset[0]
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    
    return data, train_mask, val_mask, test_mask