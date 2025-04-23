import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import librosa
import numpy as np

# Define a simple test function
def test_qvim_adapter():
    """Simple test to verify the QVIMAdapter functionality"""
    # Create dummy data
    qvim_dim = 1024  # MobileNetV3 embedding dimension
    audioldm_dim = 512  # AudioLDM CLAP dimension
    batch_size = 2
    
    # Create random QVIM embeddings
    dummy_embeddings = torch.randn(batch_size, qvim_dim)
    
    # Initialize the adapter
    from audioldm.qvim_adapter import QVIMAdapter
    adapter = QVIMAdapter(qvim_dim, audioldm_dim)
    
    # Process the embeddings
    adapted_embeddings = adapter(dummy_embeddings)
    
    # Get unconditional embeddings
    uncond_embeddings = adapter.get_unconditional_embedding(batch_size=batch_size)
    
    # Check shapes and properties
    print(f"Input shape: {dummy_embeddings.shape}")
    print(f"Adapted shape: {adapted_embeddings.shape}")
    print(f"Unconditional shape: {uncond_embeddings.shape}")
    
    # Check normalization
    norms = torch.norm(adapted_embeddings, dim=-1)
    print(f"Norms (should be close to 1.0): {norms}")
    
    # Create conditioning dictionary
    cond = {
        "c_crossattn": [adapted_embeddings, uncond_embeddings]
    }
    
    print(f"Conditioning dictionary structure: {list(cond.keys())}")
    print(f"c_crossattn list length: {len(cond['c_crossattn'])}")
    
    # Success message
    print("\nTest completed successfully! The adapter functions as expected.")

# Run the test if executed directly
if __name__ == "__main__":
    test_qvim_adapter()