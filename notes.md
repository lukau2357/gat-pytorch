## GCN - Graph Convolutional Networks

## GAT - Graph Attention Networks

## Sparse matrix representation notes
**COO (Coordinate list)** format: Representing sparse adjacency matrix as a $(2, E)$ matrix $C$, where for each edge $C[0][i]$ is the source vertex and $C[0][j]$ is the destination vertex. Can be extended easily to any real matrix, although 3 vectors would be needed in that case to keep values themselves.

**CSR (Compressed sparse row)** format: Encodes a sparse matrix with 3 one dimensional arrays, $V, C, R$. V holds values for non-zero elements, $C$ holds column indices for non zero elements, and $R[i]$ indicates the number of non-zero elements above row $i$. Last element of $R$ is equal to the total number of non-zero elements of the matrix. Convenient, especially if matrix-vector multiplication operation is common, can be efficiently implemented using this format [link](https://www.netlib.org/utk/people/JackDongarra/etemplates/node382.html).

**LIL (List of Lists)** format: One list per row, elements of lists are tuples (column_index, value). Adjacency list is a form of LIL.