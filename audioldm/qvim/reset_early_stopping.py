#!/usr/bin/env python
import torch
import argparse

def reset_early_stopping(checkpoint_path, output_path=None, patience=None, min_delta=None):
    """
    Reset early stopping counter in a PyTorch Lightning checkpoint and optionally modify parameters.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        output_path: Optional output path. If None, overwrites the original checkpoint
        patience: Optional new patience value
        min_delta: Optional new minimum delta value
    """
    output_path = output_path or checkpoint_path
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Check if early stopping state exists - using correct structure
    if "callbacks" in checkpoint and "EarlyStopping" in checkpoint["callbacks"]:
        es_state = checkpoint["callbacks"]["EarlyStopping"]
        
        # Reset the counter
        previous_count = es_state.get("wait_count", "unknown")
        es_state["wait_count"] = 0
        print(f"Reset early stopping counter from {previous_count} to 0")
        
        # Reset stopped_epoch if present
        if "stopped_epoch" in es_state:
            es_state["stopped_epoch"] = 0
            print(f"Reset stopped_epoch to 0")
        
        # Update patience if provided
        if patience is not None:
            previous_patience = es_state.get("patience", "unknown")
            es_state["patience"] = patience
            print(f"Updated early stopping patience from {previous_patience} to {patience}")
        
        # Save the modified checkpoint
        torch.save(checkpoint, output_path)
        print(f"Saved modified checkpoint to {output_path}")
    else:
        print("No early stopping state found in checkpoint.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reset early stopping counter in a PyTorch Lightning checkpoint")
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint file")
    parser.add_argument("--output", type=str, default=None, help="Output path for modified checkpoint (default: overwrite input)")
    parser.add_argument("--patience", type=int, default=None, help="New value for early stopping patience")
    
    args = parser.parse_args()
    reset_early_stopping(args.checkpoint_path, args.output, args.patience)