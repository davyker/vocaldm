#!/usr/bin/env python
import torch
import argparse
import pprint

def examine_checkpoint(checkpoint_path):
    """
    Print the structure of a PyTorch Lightning checkpoint
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Print top-level keys
    print("\nTop-level keys in checkpoint:")
    for key in checkpoint.keys():
        print(f"- {key}")
    
    # Look for callback states
    if "callbacks" in checkpoint:
        print("\nCallback keys:")
        for key in checkpoint["callbacks"].keys():
            print(f"- {key}")
            # Print callback state
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(checkpoint["callbacks"][key])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Examine PyTorch Lightning checkpoint structure")
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint file")
    
    args = parser.parse_args()
    examine_checkpoint(args.checkpoint_path)