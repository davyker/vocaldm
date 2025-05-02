import torch

# Generate 5 random vectors uniformly in [-1,1]^512
vectors = 2 * torch.rand(5, 512) - 1

# Normalize vectors
normalized = vectors / torch.norm(vectors, dim=1, keepdim=True)

# Print stats
for i, v in enumerate(normalized):
    print(f"Vector {i}: min={v.min().item():.4f}, max={v.max().item():.4f}, mean={v.mean().item():.4f}, std={v.std().item():.4f}")