#!/usr/bin/env python
"""
Extract CLAP model weights from an AudioLDM checkpoint.
This script loads an AudioLDM model and extracts the CLAP model weights.
These can then be used for training the QVIM-CLAP alignment model.
"""

import os
import argparse
import torch
from audioldm.utils import default_audioldm_config, get_metadata, CACHE_DIR
from audioldm.pipeline import build_model

def extract_clap_checkpoint(audioldm_model_name="audioldm-m-full", save_path=None):
    """
    Extract CLAP model weights from an AudioLDM checkpoint.
    
    Args:
        audioldm_model_name: Name of the AudioLDM model to extract from
        save_path: Path to save the CLAP checkpoint. If None, uses default path.
    
    Returns:
        Path to the saved CLAP checkpoint
    """
    print(f"Extracting CLAP model from {audioldm_model_name}")
    
    # Build the AudioLDM model
    model = build_model(model_name=audioldm_model_name)
    
    # Get the CLAP model from the conditioning stage
    clap_model = model.cond_stage_model
    
    # Get the appropriate filename for the CLAP model
    # Only medium models use HTSAT-base according to utils.py
    amodel_name = "HTSAT-base" if "-m-" in audioldm_model_name else "HTSAT-tiny"
    
    if save_path is None:
        # Create default path if not provided
        os.makedirs(os.path.join(CACHE_DIR, "clap"), exist_ok=True)
        save_path = os.path.join(CACHE_DIR, "clap", f"{amodel_name.lower()}.pt")
    
    # Extract CLAP model state_dict
    clap_state_dict = clap_model.model.state_dict()
    
    # Save the CLAP model state_dict
    torch.save(clap_state_dict, save_path)
    
    print(f"CLAP model extracted from {audioldm_model_name}")
    print(f"CLAP model architecture: {amodel_name}")
    print(f"CLAP checkpoint saved to {save_path}")
    
    return save_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CLAP model from AudioLDM checkpoint")
    parser.add_argument("--model", type=str, default="audioldm-m-full", 
                        help="AudioLDM model name (audioldm-s-full, audioldm-m-full, audioldm-l-full)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for CLAP checkpoint. If not provided, uses default cache location.")
    
    args = parser.parse_args()
    
    extract_clap_checkpoint(args.model, args.output)