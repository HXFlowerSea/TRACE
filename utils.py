from typing import Dict, List, Tuple
import torch
from torch_geometric.data import Data
from torch_geometric import transforms as T
import pickle
import dgl
import numpy as np
import scipy.sparse as sp
import networkx as nx
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def load_data(dataset, task_cls):
    subgraph, ids_per_cls_all, [train_ids, valid_ids_, test_ids_] = pickle.load(open(
            f'./data/{dataset}_{task_cls}.pkl', 'rb')) # CoraFull-CL_[0, 1]
    subgraph = dgl.add_self_loop(subgraph)
    subgraph_adj = build_symmetric(subgraph.adj(scipy_fmt='csr'))
    features = sp.lil_matrix(preprocess_features(subgraph.srcdata['feat'].numpy()))
    features = preprocess_features(features)
    labels = subgraph.dstdata['label'].squeeze().numpy()
    return subgraph_adj, features, labels, ids_per_cls_all, [train_ids, valid_ids_, test_ids_]

def data_to_pyg(adj, features, labels, ids_per_cls_all, train_ids, valid_ids_, test_ids_):
    row, col = adj.nonzero()
    row = np.array(row) 
    col = np.array(col) 
    edge_index_np = np.stack([row, col], axis=0)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    x = torch.tensor(features.toarray(), dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    num_nodes = x.size(0)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_ids] = True
    val_mask[valid_ids_] = True
    test_mask[test_ids_] = True
    
    masks = [{
        'train': train_mask,
        'val': val_mask,
        'test': test_mask
    }]
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data, masks
    

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    #return sparse_to_tuple(features)
    return features#.todense()

def build_symmetric(adj):
    adj = adj.tocoo()
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj.tocsr()

def cos_sim(tensor_1, tensor_2):
    normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
    normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)

    return torch.mm(normalized_tensor_1, torch.t(normalized_tensor_2))

def compute_mse(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        raise ValueError("two matrix must be same shape！")
    mse = torch.mean((matrix1 - matrix2) ** 2)
    return mse.item() 

def compute_avg(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        raise ValueError("two matrix must be same shape！")
    mean_error = torch.mean(matrix1 - matrix2)
    return mean_error.item() 

def compute_mae(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        raise ValueError("two matrix must be same shape！")
    mae = torch.mean(torch.abs(matrix1 - matrix2))
    return mae.item() 