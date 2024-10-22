import torch
from typing import List, Tuple

NONLINEARITY_DICT = {
    "elu": torch.nn.ELU(),
    "identity": torch.nn.Identity()
}

class GATLayer(torch.nn.Module):
    def __init__(self, input_dim : int, output_dim : int, attention_heads : int,
                 leaky_relu_alpha : float = 0.2, 
                 dropout_p : float = 0, 
                 concat_last : bool = True,
                 nonlinearity = "elu",
                 residual : bool = True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.attention_heads = attention_heads
        self.num_hidden_layers = 1
        self.leaky_relu_alpha = leaky_relu_alpha
        self.dropout_p = dropout_p
        self.concat_last = concat_last
        self.nonlinearity = nonlinearity
        self.residual = residual

        # Create attention mask from adjacency matrix.
        self.W = torch.nn.Parameter(torch.zeros((input_dim, attention_heads * output_dim), dtype = torch.float32))
        self.A_left = torch.nn.Parameter(torch.zeros((attention_heads, output_dim, 1), dtype = torch.float32))
        self.A_right = torch.nn.Parameter(torch.zeros((attention_heads, output_dim, 1), dtype = torch.float32))

        self.residual_projection = None
        # If input and output dimensions allow addition, no need to apply seperate mapping, save on computation and number of parameters
        if self.residual:
            self.residual_projection = torch.nn.Parameter(torch.zeros((input_dim, attention_heads * output_dim), dtype = torch.float32))

        self.lrelu = torch.nn.LeakyReLU(negative_slope = leaky_relu_alpha)
        self.dropout = torch.nn.Dropout(p = self.dropout_p)
        
        # TODO: More elegant way to write this, instead of if statement for every key
        self.nonlinearity_layer = NONLINEARITY_DICT[self.nonlinearity]
        self._init_weights()
        
        # For visualization purposes
        self.attention_weights = None
    def _init_weights(self):
        # TODO: In my opinion, initialization should be applied seperately for each attention head, rather than parameterizing 
        # shapes self.W, self.A_left, self.A_right. I did not however want to deviate from authors that much, although they use Xavier uniform init instead.
        torch.nn.init.kaiming_normal_(self.W, a = self.leaky_relu_alpha)
        torch.nn.init.kaiming_normal_(self.A_left, a = self.leaky_relu_alpha)
        torch.nn.init.kaiming_normal_(self.A_right, a = self.leaky_relu_alpha)

        if self.residual_projection is not None:
            torch.nn.init.kaiming_normal_(self.residual_projection, a = self.leaky_relu_alpha)
    
    def forward(self, data : Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_proj = data[0]
        edge_index = data[1]

        input_proj = self.dropout(input_proj)
        N = input_proj.shape[0]
        H = torch.matmul(input_proj, self.W).view((N, self.attention_heads, -1)).permute((1, 0, 2)) # [attention_heads, N, output_dim]
        H_s = torch.matmul(H, self.A_left).squeeze(-1) # [attention_heads, N]
        H_t = (torch.matmul(H, self.A_right)).squeeze(-1) # [attention_heads, N]

        H_sf, H_ss, H_tt = self._select_scores(H, H_s, H_t, edge_index)
        edge_scores = self.lrelu(H_ss + H_tt) # [attention_heads, E]

        # Compute attention scores accross all attention heads for each node.
        edge_scores_norm = self._topological_softmax(edge_scores, N, edge_index[0]) # [attention_heads, E]
        edge_scores_norm = self.dropout(edge_scores_norm) # [attention_heads, E]
        self.attention_weights = edge_scores_norm.detach() # Save attention weights, assumption is that the model has been put in evaluation mode before this!
        per_head_mappings = self._edge_scores_norm_to_per_head_mappings(edge_scores_norm, H_sf, N, edge_index[0]) # [attention_heads, N, output_dim]

        if self.residual:
            if per_head_mappings.shape[-1] != input_proj.shape[-1]:
                per_head_mappings += torch.matmul(input_proj, self.residual_projection).view((N, self.attention_heads, -1)).permute((1, 0, 2))

            else:
                per_head_mappings += input_proj.unsqueeze(0)

        if self.concat_last:
            res = per_head_mappings.permute(1, 0, 2).contiguous().view(N, self.attention_heads * self.output_dim) # [N, attention_heads * output_dim]
        
        else:
            res = per_head_mappings.mean(dim = 0) # [N, output_dim]
        
        # For concatenation, according to equations (5) in https://arxiv.org/pdf/1710.10903, authors apply nonlinearity before concatenating embeddings accross 
        # different attention heads. For simplicity, we apply nonlinearity only at the final step, whether mean reduction or concatenation is used.
        return self.nonlinearity_layer(res), edge_index

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
    
class GAT(torch.nn.Module):
    def __init__(self, input_dim : int, heads_per_layer : List[int], features_per_layer : List[int], 
                 residual : bool = True, 
                 dropout_p : float = 0.6, 
                 nonlinearity : str = "elu"):
        super().__init__()
        self.input_dim = input_dim
        self.heads_per_layer = heads_per_layer
        self.features_per_layer = features_per_layer
        self.residual = residual
        self.dropout_p = dropout_p
        self.nonlinearity = nonlinearity

        layers = [GATLayer(input_dim, features_per_layer[0], 
                           attention_heads = heads_per_layer[0], 
                           dropout_p = dropout_p, 
                           concat_last = len(heads_per_layer) > 1, 
                           residual = residual,
                           nonlinearity = "identity" if len(heads_per_layer) == 1 else nonlinearity)]
        
        for i in range(1, len(heads_per_layer)):
            layers.append(GATLayer(heads_per_layer[i - 1] * features_per_layer[i - 1], features_per_layer[i],
                                   attention_heads = heads_per_layer[i],
                                   dropout_p = dropout_p,
                                   concat_last = i < len(heads_per_layer) - 1,
                                   residual = residual,
                                   nonlinearity = "identity" if i == len(heads_per_layer) - 1 else nonlinearity))
        
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, data : Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # Omit edge index from forward pass result
        return self.layers.forward(data)[0]
    
    def to_dict(self):
        d = {
            "input_dim": self.input_dim,
            "heads_per_layer": self.heads_per_layer,
            "features_per_layer": self.features_per_layer,
            "residual": self.residual,
            "dropout_p": self.dropout_p,
            "nonlinearity": self.nonlinearity,
        }

        return d
    
    @classmethod
    def from_dict(cls, d):
        return cls(d["input_dim"], d["heads_per_layer"], d["features_per_layer"],
                    residual = d["residual"], 
                    dropout_p = d["dropout_p"], 
                    nonlinearity = d["nonlinearity"])