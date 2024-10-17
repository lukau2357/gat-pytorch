import torch

if __name__ == "__main__":
    torch.manual_seed(41)
    target = torch.zeros((6), dtype = torch.long)

    source_vertices = torch.randint(low = 0, high = 6, size = (15,), dtype = torch.long)
    weights = torch.randint(low = 0, high = 100, size = (15,), dtype = torch.long)

    print(weights)
    print(source_vertices)
    target.scatter_add_(0, source_vertices, weights)
    print(target)