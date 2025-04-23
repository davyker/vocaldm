import torch
import pprint

# Path to checkpoint
checkpoint_path = "audioldm/qvim/models_vimsketch_longer/dulcet-leaf-31/best-loss-checkpoint.ckpt"

# Load the checkpoint
print(f"Loading checkpoint from {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Print the top-level keys
print("\nCheckpoint top-level keys:")
print(list(checkpoint.keys()))

# Print the first level of nested structure
pp = pprint.PrettyPrinter(indent=2, depth=2)
print("\nCheckpoint structure (limited depth):")
pp.pprint(checkpoint)

# If 'state_dict' exists, print some of the model keys
if 'state_dict' in checkpoint:
    print("\nSome model keys:")
    # Get the first few keys
    model_keys = list(checkpoint['state_dict'].keys())[:5]
    for key in model_keys:
        print(f"  {key}")