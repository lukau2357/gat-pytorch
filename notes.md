cora.cites direction is right to left. For instance, if there is a line "paper1" "paper2", then this implies the edge "paper2" -> "paper1".

COO (Coordinate list) format: Representing sparse adjacency matrix as a $(2, E)$ matrix $C$, where for each edge $C[0][i]$ is the source vertex and $C[0][j]$ is the destination vertex.

CSR (Compressed sparse row) format: Encodes a sparse matrix with 3 one dimensional arrays, $V, C, R$. V holds values for non-zero elements, $C$ holds column indices for non zero elements, and $R[i]$ indicates the number of non-zero elements above row $i$. Last element of $R$ is equal to the total number of non-zero elements of the matrix.

LIL (List of Lists) format: One list per row, elements of lists are tuples (column_index, value).

# TODO
- Add visualizations for degree distribution, etc.
- Add Tensorboard logging
- Implement patience mechanism perhaps? They did that in the paper, but even without it results were reproduced.