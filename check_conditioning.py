#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import os
import sys
from audioldm import build_model
from audioldm.pipeline import make_batch_for_text_to_audio

# Configure model
model_name = "audioldm-m-full"
print(f"Loading AudioLDM model: {model_name}")
audioldm = build_model(model_name=model_name)

# Create a text conditioning batch
text = "Dog barking"
batch = make_batch_for_text_to_audio(text, batchsize=1)

# Get the model input
print("Getting input from batch...")
z, c = audioldm.get_input(batch, audioldm.first_stage_key, cond_key=audioldm.cond_stage_key)

# Analyze the conditioning
print("\nConditioning Analysis:")
print(f"Type of conditioning: {type(c)}")

if isinstance(c, torch.Tensor):
    print(f"Shape: {c.shape}")
    print(f"Device: {c.device}")
    print(f"Data type: {c.dtype}")
    print(f"First few values: {c[0, :5]}")
elif isinstance(c, list):
    print(f"List length: {len(c)}")
    for i, item in enumerate(c):
        print(f"  Item {i} type: {type(item)}")
        if isinstance(item, torch.Tensor):
            print(f"  Item {i} shape: {item.shape}")
            print(f"  Item {i} device: {item.device}")
            print(f"  Item {i} data type: {item.dtype}")
            print(f"  Item {i} first few values: {item[0, :5]}")
elif isinstance(c, dict):
    print(f"Dict keys: {c.keys()}")
    for key, value in c.items():
        print(f"  Key: {key}, Value type: {type(value)}")
        if isinstance(value, torch.Tensor):
            print(f"  Value shape: {value.shape}")
        elif isinstance(value, list):
            print(f"  List length: {len(value)}")
            for i, item in enumerate(value):
                print(f"    Item {i} type: {type(item)}")
                if isinstance(item, torch.Tensor):
                    print(f"    Item {i} shape: {item.shape}")

# Try to generate sample
print("\nAttempting to generate a sample...")
try:
    samples, _ = audioldm.sample_log(
        cond=c,
        batch_size=1,
        ddim=True,
        ddim_steps=50,
        unconditional_guidance_scale=3.0
    )
    print("Sample generation successful!")
    print(f"Sample shape: {samples.shape}")
except Exception as e:
    print(f"Sample generation failed: {e}")
    
    # Try to fix the conditioning based on error
    if "list" in str(e) and "shape" in str(e):
        print("\nTrying to fix conditioning format...")
        if isinstance(c, list) and len(c) > 0:
            c_fixed = c[0]
            print(f"Extracted tensor from list, shape: {c_fixed.shape}")
            
            try:
                samples, _ = audioldm.sample_log(
                    cond=c_fixed,
                    batch_size=1,
                    ddim=True,
                    ddim_steps=50,
                    unconditional_guidance_scale=3.0
                )
                print("Sample generation with fixed conditioning successful!")
                print(f"Sample shape: {samples.shape}")
            except Exception as e2:
                print(f"Sample generation with fixed conditioning failed: {e2}")