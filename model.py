import torch
from utils import load_data

class GATLayer(torch.nn.Module):
    def __init__(self, input_dim : int, output_dim : int, attention_heads : int,
                 leaky_relu_alpha : float = 0.2, 
                 dropout_p : float = 0, 
                 concat_last : bool = True,
                 nonlinearity = "elu"):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.attention_heads = attention_heads
        self.num_hidden_layers = 1
        self.leaky_relu_alpha = leaky_relu_alpha
        self.dropout_p = dropout_p
        self.concat_last = concat_last
        self.nonlinearity = nonlinearity

        # Create attention mask from adjacency matrix.
        self.W = torch.nn.Parameter(torch.zeros((input_dim, attention_heads * output_dim), dtype = torch.float32))
        self.A_left = torch.nn.Parameter(torch.zeros((attention_heads, output_dim, 1), dtype = torch.float32))
        self.A_right = torch.nn.Parameter(torch.zeros((attention_heads, output_dim, 1), dtype = torch.float32))

        self.lrelu = torch.nn.LeakyReLU(negative_slope = leaky_relu_alpha)
        self.attention_dropout = torch.nn.Dropout(p = self.dropout_p)
        
        # TODO: More elegant way to write this, instead of if statement for every key
        if self.nonlinearity == "elu":
            self.nonlinearity_layer = torch.nn.ELU()
        
        else:
            self.nonlinearity_layer = torch.nn.Identity()

    def _init_weights(self):
        # Manual Kaiming Normal fan_in initialization, since we want all attention head linear maps to be initialized independently.
        leaky_relu_gain = (1 / (1 + self.leaky_relu_alpha))
        torch.nn.init.normal_(self.W, mean = 0, std = (leaky_relu_gain / self.input_dim) ** 0.5)
        torch.nn.init.normal_(self.A_left, mean = 0, std = (leaky_relu_gain / self.output_dim))
        torch.nn.init.normal_(self.A_right, mean = 0, std = (leaky_relu_gain / self.output_dim))
    
    def forward(self, input_proj : torch.Tensor, edge_index : torch.Tensor) -> torch.Tensor:
        N = input_proj.shape[0]
        H = torch.matmul(input_proj, self.W).view((N, self.attention_heads, -1)).permute((1, 0, 2)) # [attention_heads, N, output_dim]
        H_s = torch.matmul(H, self.A_left).squeeze(-1) # [attention_heads, N]
        H_t = (torch.matmul(H, self.A_right)).squeeze(-1) # [attention_heads, N]

        H_sf, H_ss, H_tt = self._select_scores(H, H_s, H_t, edge_index)
        edge_scores = self.lrelu(H_ss + H_tt) # [attention_heads, E]

        # Compute attention scores accross all attention heads for each node.
        edge_scores_norm = self._topological_softmax(edge_scores, N, edge_index[0]) # [attention_heads, E]
        edge_scores_norm = self.attention_dropout(edge_scores_norm) # [attention_heads, E]
        per_head_mappings = self._edge_scores_norm_to_per_head_mappings(edge_scores_norm, H_sf, N, edge_index[0]) # [attention_heads, N, output_dim]

        if self.concat_last:
            res = per_head_mappings.permute(1, 0, 2).contiguous().view(N, self.attention_heads * self.output_dim) # [N, attention_heads * output_dim]
        
        else:
            res = per_head_mappings.mean(dim = 0) # [N, output_dim]
        
        # For concatenation, according to equations (5) in https://arxiv.org/pdf/1710.10903, authors apply nonlinearity before concatenating embeddings accross 
        # different attention heads. For simplicity, we apply nonlinearity only at the final step, whether mean reduction or concatenation is used.
        return self.nonlinearity_layer(res)

    def _select_scores(self, H : torch.Tensor, H_s : torch.Tensor, H_t : torch.Tensor, edge_index : torch.Tensor):
        source_index = edge_index[0]
        target_index = edge_index[1]

        H_ss = H_s.index_select(1, source_index) # [attention_heads, E], E is the number of edges, including added loops
        H_tt = H_t.index_select(1, target_index) # [attention_heads, E]
        H_sf = H.index_select(1, target_index)   # [attention_heads, E, output_dim]

        return H_sf, H_ss, H_tt 

    def _topological_softmax(self, edge_scores : torch.Tensor, N : int, source_index : torch.Tensor):
        edge_scores = edge_scores - edge_scores.max() # Numerically more stable logits for sofmtax
        edge_scores = edge_scores.exp() # [attention_heads, E]

        attention_heads = edge_scores.shape[0]
        neighborhood_sums = torch.zeros((attention_heads, N), dtype = torch.float32, device = edge_scores.device)
        source_index_brd = torch.broadcast_to(source_index, edge_scores.shape)
        # https://pytorch.org/docs/stable/generated/torch.scatter_add.html
        neighborhood_sums = neighborhood_sums.scatter_add_(1, source_index_brd, edge_scores) # [attention_heads, N]
        neighborhood_sums = neighborhood_sums.index_select(1, source_index) # [attention_heads, E]
        return edge_scores / (neighborhood_sums + 1e-8) # [attention_heads, E], 1e-8 for numerical stability

    def _edge_scores_norm_to_per_head_mappings(self, edge_scores_norm : torch.Tensor, H_sf : torch.Tensor, N : int, source_index : torch.Tensor):
        attention_heads = edge_scores_norm.shape[0]
        attention_prod = edge_scores_norm.unsqueeze(-1) * H_sf # [attention_heads, E, output_dim]
        res = torch.zeros((attention_heads, N, self.output_dim), device = edge_scores_norm.device)
        source_index_brd = torch.broadcast_to(source_index.unsqueeze(0).unsqueeze(-1), attention_prod.shape) # source_index is of shape [E] so unsqueezing is required to broadcast it properly
        res.scatter_add_(1, source_index_brd, attention_prod)
        return res # [attention_heads, N, output_dim]
    
if __name__ == "__main__":
    path = r"C:\Users\Korisnik\Desktop\gnn\cora"
    device = "cuda"
    feature_matrix, node_labels, edge_index = load_data(path, device)
    input_dim = feature_matrix.shape[1]
    layer = GATLayer(input_dim, 8, attention_heads = 8).to(device)
    print(layer.forward(feature_matrix, edge_index).shape)