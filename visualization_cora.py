import torch
import argparse
import os
import json
import matplotlib.pyplot as plt
import time
import numpy as np

from model import GAT
from utils import load_data
from sklearn.manifold import TSNE

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("data_dir", type = str, help = "Path to the CORA dataset.")
    args.add_argument("model_dir", type = str, help = "Path to the GAT CORA checkpoint.")
    args.add_argument("output_dir", type = str, help = "Path to the directory that will contain output figures.")
    args.add_argument("--tsne_perplexity", type = float, default = 30.0, help = "Specify perplexity to be used by t-SNE. Perplexity is related to the number of nearest neighbors in the input embedding space.")
    args.add_argument("--random_seed", type = int, default = 41, help = "Specify reproducibility seed.")
    args.add_argument("--marker_size", type = float, default = 40.0, help = "Specify area of a single point on the graph embedding plot.")
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

    for i in range(edge_attentions.shape[1]):
        source_vertex = edge_index_np[0][i]
        target_vertex = edge_index_np[1][i]

        if source_vertex == target_vertex:
            attentions[source_vertex] = edge_attentions[0][i].item()
    
    vertices = sorted(attentions.keys())
    return [attentions[vertex] for vertex in vertices]

if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    feature_matrix, node_labels, edge_index = load_data(args.data_dir, device)

    model = from_checkpoint(args.model_dir, feature_matrix.shape[-1]).to(device)
    model.eval()

    with torch.no_grad():
        gat_embeddings = model.forward((feature_matrix, edge_index)).cpu().numpy()
        edge_attentions = model.layers[1].attention_weights
        
    node_labels_np = node_labels.cpu().numpy()
    edge_index_np = edge_index.cpu().numpy()

    tsne = TSNE(n_components = 2, learning_rate = "auto", perplexity = args.tsne_perplexity, random_state = args.random_seed)
    start = time.time()
    print("Fitting 2 dimensional t-SNE to CORA GAT embeddings.")
    gat_embeddings_tsne = tsne.fit_transform(gat_embeddings)
    print(f"t-SNE training time: {(time.time() - start):.4f}s.")

    cmap = plt.get_cmap("Paired", 7)
    class_labels = np.array(list(range(7)))
    colors = cmap(np.arange(7))
    class_color_map = {cls: colors[i] for i, cls in enumerate(class_labels)}
    color_array = np.array([class_color_map[cls] for cls in node_labels_np])

    fig, ax = plt.subplots()

    if edge_attentions.shape[0] > 1:
        print("Impossible to transparently visualize edge attentions, last layer has multiple attention heads.")
        ax.scatter(gat_embeddings_tsne[:, 0], gat_embeddings_tsne[:, 1], 
            s = args.marker_size, 
            c = color_array, 
            edgecolors = [(0.5, 0.5, 0.5, edge_attentions[0][i]) for i in range()], 
            linewidth = 0.3)
    else:
        self_attentions = extract_self_attentions(edge_index_np, edge_attentions)
        print("Extracted self attentions")

        ax.scatter(gat_embeddings_tsne[:, 0], gat_embeddings_tsne[:, 1], 
            s = args.marker_size, 
            c = color_array, 
            edgecolors = [(0.75, 0.75, 0.75, alpha) for alpha in self_attentions], 
            linewidth = 2.5)

    if edge_attentions.shape[0] == 1:
        print("Visualizing edge attentions in the last GAT layer")
        start = time.time()
        attentions = {}

        for i in range(edge_index_np.shape[1]):
            source_vertex = edge_index_np[0][i]
            target_vertex = edge_index_np[1][i]
            key = (min(source_vertex, target_vertex), max(source_vertex, target_vertex))
            # CORA is viewed as an undirected graph, and if we consider two directions of an edge (i, j), (j, i), assigned attentions could be different
            # For the purposes of visualization, we will take the maximum of attentions for both directions of an edge.
            attentions[key] = max(attentions.get(key, -1), edge_attentions[0][i].item())

        for key, value in attentions.items():
            source_vertex = key[0]
            target_vertex = key[1]
            attention_score = value
            
            # Edge attention
            if source_vertex != target_vertex:
                max_x = max(gat_embeddings_tsne[source_vertex][0], gat_embeddings_tsne[target_vertex][0])
                min_x = min(gat_embeddings_tsne[source_vertex][0], gat_embeddings_tsne[target_vertex][0])
                max_y = max(gat_embeddings_tsne[source_vertex][1], gat_embeddings_tsne[target_vertex][1])
                min_y = min(gat_embeddings_tsne[source_vertex][1], gat_embeddings_tsne[target_vertex][1])

                x_values = [min_x + args.marker_size / 2, max_x - args.marker_size / 2]
                y_values = [min_y + args.marker_size / 2, max_y - args.marker_size / 2]

                ax.plot(x_values, y_values, color = "gray", linewidth = 0.6, alpha = attention_score)
        
        print(f"Done visualizing edge attentions. Processing time: {(time.time() - start):.4f}s.")

    ax.axis("off")
    plt.show()
    # fig.savefig(os.path.join(args.output_dir, "cora_embeddings.png"))