import torch
import argparse
import json
import matplotlib.pyplot as plt
import time
import numpy as np
import os

from model import GAT
from utils import load_data_cora
from sklearn.manifold import TSNE

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("data_dir", type = str, help = "Path to the CORA dataset.")
    args.add_argument("model_dir", type = str, help = "Path to the directory which contains CORA GAT checkpoint.")
    args.add_argument("--tsne_perplexity", type = float, default = 30.0, help = "Specify perplexity to be used by t-SNE. Perplexity is related to the number of nearest neighbors in the input embedding space.")
    args.add_argument("--marker_size", type = float, default = 40.0, help = "Specify area of a single point on the graph embedding plot.")
    args.add_argument("--attention_layer_index", type = int, default = 0, help = "Specify the index of GAT layer for which attention coefficients will be taken for edge visualization.")
    return args.parse_args()

def from_checkpoint(model_dir : str, input_features : int) -> GAT:
    with open(os.path.join(model_dir, ".metadata.json"), "r", encoding = "utf-8") as f:
        metadata = json.load(f)
    
    model = GAT(input_features,
                heads_per_layer = metadata["heads_per_layer"],
                features_per_layer = metadata["features_per_layer"],
                dropout_p = metadata["dropout_p"],
                residual = metadata["residual"])
    
    model_state_dict = torch.load(os.path.join(model_dir, "checkpoint.pth"), weights_only = True)
    model.load_state_dict(model_state_dict)
    return model

def extract_self_attentions(edge_index_np, edge_attentions):
    attentions = {}

    for i in range(edge_attentions.shape[0]):
        source_vertex = edge_index_np[0][i]
        target_vertex = edge_index_np[1][i]

        if source_vertex == target_vertex:
            attentions[source_vertex] = edge_attentions[i].item()
    
    vertices = sorted(attentions.keys())
    return [attentions[vertex] for vertex in vertices]

if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_matrix, node_labels, edge_index = load_data_cora(args.data_dir, device)

    model = from_checkpoint(args.model_dir, feature_matrix.shape[-1]).to(device)
    model.eval()

    with torch.no_grad():
        gat_embeddings = model.forward((feature_matrix, edge_index)).cpu().numpy()
        edge_attentions = model.layers[args.attention_layer_index].attention_weights
        
    node_labels_np = node_labels.cpu().numpy()
    edge_index_np = edge_index.cpu().numpy()

    tsne = TSNE(n_components = 2, learning_rate = "auto", perplexity = args.tsne_perplexity)
    start = time.time()
    print("Fitting 2 dimensional t-SNE to CORA GAT embeddings.")
    gat_embeddings_tsne = tsne.fit_transform(gat_embeddings)
    print(f"t-SNE training time: {(time.time() - start):.4f}s.")
    
    class_color_map = {0: "#0205FD", 1: "#FD66AA", 2: "#008002", 3: "#FDA401", 4: "#FC01FD", 5: "#54BAFD", 6: "#FD0203"}
    color_array = np.array([class_color_map[cls] for cls in node_labels_np])
    fig, ax = plt.subplots()

    print(f"Visualizing edge attentions in {args.attention_layer_index}-th GAT layer.")
    start = time.time()
    # Sum attentions for particular edge accross all attention heads
    edge_attentions = edge_attentions.sum(dim = 0)
    # Topological softmax on edge attentions, to be used for edge thicnkess in visualization
    edge_attentions = (edge_attentions - edge_attentions.max()).exp()
    edge_attentions_accum = torch.zeros((feature_matrix.shape[0]), dtype = torch.float32, device = device)
    edge_attentions_accum = edge_attentions_accum.scatter_add_(0, edge_index[0], edge_attentions)
    edge_attentions_accum = edge_attentions_accum.index_select(0, edge_index[0])
    edge_attentions = edge_attentions / edge_attentions_accum

    attentions = {}

    for i in range(edge_index_np.shape[1]):
        source_vertex = edge_index_np[0][i]
        target_vertex = edge_index_np[1][i]
        key = (min(source_vertex, target_vertex), max(source_vertex, target_vertex))
        # Add attention coefficients for a single edge in both directions
        attentions[key] = attentions.get(key, 0) + edge_attentions[i].item()

    for key, value in attentions.items():
        source_vertex = key[0]
        target_vertex = key[1]
        attention_score = value
        
        # Edge attention
        if source_vertex != target_vertex:
            x_values = [gat_embeddings_tsne[source_vertex][0], gat_embeddings_tsne[target_vertex][0]]
            y_values = [gat_embeddings_tsne[source_vertex][1], gat_embeddings_tsne[target_vertex][1]]
            color_val = max(0, 1 - attention_score)
            ax.plot(x_values, y_values, 
                    color = (color_val, color_val, color_val), 
                    linewidth = 0.2)
    
    print(f"Done visualizing edge attentions. Processing time: {(time.time() - start):.4f}s.")

    print(f"Visualizing self attentions in {args.attention_layer_index}-th GAT layer.")
    start = time.time()
    self_attentions = extract_self_attentions(edge_index_np, edge_attentions)
    
    ax.scatter(gat_embeddings_tsne[:, 0], gat_embeddings_tsne[:, 1], 
        s = args.marker_size, 
        c = color_array,
        edgecolors = [(max(0, 1 - alpha), max(0, 1 - alpha), max(0, 1 - alpha)) for alpha in self_attentions], 
        linewidth = 1.5)
    
    print(f"Done visualizing self attentions. Processing time: {(time.time() - start):.4f}s.")
    ax.axis("off")
    plt.show()