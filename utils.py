import os
import numpy as np
import scipy.sparse as sp
import torch
import pickle

CLASS_TO_ID = {
		"Case_Based": 0,
		"Genetic_Algorithms": 1,
		"Neural_Networks": 2,
		"Probabilistic_Methods": 3,
		"Reinforcement_Learning": 4,
		"Rule_Learning": 5,
		"Theory": 6
}

# Train/Val/Test split according to GAT/GCN authors
CORA_TRAIN_RANGE = [0, 140]  # we're using the first 140 nodes as the training nodes
CORA_VAL_RANGE = [140, 140 + 500]
CORA_TEST_RANGE = [1708, 1708 + 1000]

def normalize_feature_matrix(x):
    if sp.issparse(x):
        sum_features = np.array(x.sum(-1, dtype = float)).squeeze(-1)
        sum_features = np.power(sum_features, -1)
        sum_features[np.isinf(sum_features)] = 1.0 # Zero sum of features could lead to zero division
        return sp.diags(sum_features).dot(x)
    
    # TODO: Consider dense feature matrix, should be easier
    else:
        pass

def load_data(path, device):
    with open(os.path.join(path, "node_features.csr"), "rb") as f:
        feature_matrix = pickle.load(f, encoding = "latin1")

    feature_matrix = sp.csr_matrix(feature_matrix)

    with open(os.path.join(path, f"node_labels.npy"), "rb") as f:
        node_labels = pickle.load(f, encoding = "latin1")   

    with open(os.path.join(path, "adjacency_list.dict"), "rb") as f:
        adj_dict = pickle.load(f)

    feature_matrix = normalize_feature_matrix(feature_matrix)
    
    # TODO: Use sparse feature matrix as well?
    feature_matrix = torch.tensor(feature_matrix.todense(), device = device, dtype = torch.float32)
    node_labels = torch.tensor(node_labels, dtype = torch.long, device = device)

    source_indices, target_indices = [], []
    edges_set = set()

    for source_vertex in adj_dict.keys():
        for target_vertex in adj_dict[source_vertex]:
            if (source_vertex, target_vertex) not in edges_set:
                source_indices.append(source_vertex)
                target_indices.append(target_vertex)
                edges_set.add((source_vertex, target_vertex))
                
        # Add loops, if they already don't exist
        # Attention allows a vertex to attend to itself!
        # TODO: adj_dict[source_vertex] is a list, for huge graphs could be time consuming. Optimize?
        if source_vertex not in adj_dict[source_vertex]:
            source_indices.append(source_vertex)
            target_indices.append(source_vertex)
        
    edge_index = torch.tensor(np.stack([np.array(source_indices), np.array(target_indices)], axis = 0), device = device, dtype = torch.long)
    return feature_matrix, node_labels, edge_index