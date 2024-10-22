import os
import numpy as np
import scipy.sparse as sp
import torch
import pickle
import networkx as nx
import json

from networkx.readwrite import json_graph
from torch.utils.data import DataLoader, Dataset

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

def load_data_cora(path, device):
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

def accuracy(pred : torch.Tensor, true : torch.Tensor) -> float:
    return ((torch.eq(torch.argmax(pred, dim = -1), true)).sum().item()) / true.shape[0]

class PPIDataLoader(DataLoader):
    def __init__(self, node_features_list, node_labels_list, edge_index_list, batch_size = 1, shuffle = True):
        graph_dataset = PPIDataset(node_features_list, node_labels_list, edge_index_list)
        # We need to specify a custom collate function, it doesn't work with the default one
        super().__init__(graph_dataset, batch_size, shuffle, collate_fn = graph_collate_fn_ppi)

class PPIDataset(Dataset):
    def __init__(self, node_features_list, node_labels_list, edge_index_list):
        self.node_features_list = node_features_list
        self.node_labels_list = node_labels_list
        self.edge_index_list = edge_index_list

    def __len__(self):
        return len(self.edge_index_list)

    def __getitem__(self, idx):
        # Returns all data related to graph at idx: feature matrix, node labels and edge index
        return self.node_features_list[idx], self.node_labels_list[idx], self.edge_index_list[idx]

def graph_collate_fn_ppi(batch):
    edge_index_list = []
    node_features_list = []
    node_labels_list = []
    num_nodes_seen = 0

    for features_labels_edge_index_tuple in batch:
        # Just collect these into separate lists
        node_features_list.append(features_labels_edge_index_tuple[0])
        node_labels_list.append(features_labels_edge_index_tuple[1])

        edge_index = features_labels_edge_index_tuple[2]  # In range [0, N]
        edge_index_list.append(edge_index + num_nodes_seen)  # Translates by the number of previous nodes, resulting edge_index represents different connected components
        num_nodes_seen += len(features_labels_edge_index_tuple[1])

    node_features = torch.cat(node_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0)
    edge_index = torch.cat(edge_index_list, 1)
    return node_features, node_labels, edge_index

def load_ppi_partition(data_dir : str, partition : str, batch_size : int = 1):
    print(f"Loading partition: {partition}")

    with open(os.path.join(data_dir, f"{partition}_graph.json"), "r", encoding = "utf-8") as f:
        json_graphs = json.load(f)

    with open(os.path.join(data_dir, f"{partition}_feats.npy"), "rb") as f:
        features = np.load(f)

    with open(os.path.join(data_dir, f"{partition}_graph_id.npy"), "rb") as f:
        graph_id = np.load(f)

    with open(os.path.join(data_dir, f"{partition}_labels.npy"), "rb") as f:
        labels = np.load(f)

    graph_id_min = np.min(graph_id)
    graph_id_max = np.max(graph_id)
    # NetworkX DirectedGraph class, useful.
    graphs = nx.DiGraph(json_graph.node_link_graph(json_graphs))
    num_graphs = graph_id_max - graph_id_min + 1
    num_features = features.shape[-1]
    num_classes = labels.shape[-1]

    print(f"Number of graphs in partition {partition}: {num_graphs}")

    features_list, labels_list, edge_index_list = [], [], []

    for i in range(graph_id_min, graph_id_max + 1):
        mask = graph_id == i
        # Collect all nodes that belong to current graph.
        # It apppears that the dataset is ordered so that node_ids at every range represent an arithmetic progression with step size of 1
        # This is convenient for squishing node labels to [0, N - 1], where N is the number of nodes in the subgraph generated by node_ids.

        node_ids = mask.nonzero()[0]
        subgraph = graphs.subgraph(node_ids) # Node IDS do not change when taking a subgraph

        edge_index = torch.tensor(list(subgraph.edges), dtype = torch.long).permute(1, 0)
        edge_index = edge_index - edge_index.min() # Because of batching in GraphDataLoader, it is esential that edge_index is transformed to values in [0, N - 1], where N is the number of nodes.

        # features[node_ids] -> (N, 50)
        features_list.append(torch.tensor(features[node_ids], dtype = torch.float32))
        # labels[node_ids] -> (N, 121)
        # Unlike CrossEntropyLoss, BCEWithLogits loss expects floats instead of longs -> x2 memory saved
        labels_list.append(torch.tensor(labels[node_ids], dtype = torch.float32))
        edge_index_list.append(edge_index)

    return PPIDataLoader(features_list, labels_list, edge_index_list, batch_size = batch_size), num_features, num_classes