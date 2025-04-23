#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

# Explicitly add the project directory to path (in case that helps)
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

print("Project directory added to path:", project_dir)

# Import the specific module that was failing
try:
    from audioldm.qvim_adapter import extract_qvim_embedding, prepare_vocim_conditioning
    print("Successfully imported extract_qvim_embedding from audioldm.qvim_adapter!")
except ImportError as e:
    print(f"Failed to import from audioldm.qvim_adapter: {e}")

# Try importing from other modules that would be needed in training
try:
    from audioldm.qvim.src.qvim_mn_baseline.dataset import VimSketchDataset
    from audioldm.qvim.src.qvim_mn_baseline.utils import NAME_TO_WIDTH
    print("Successfully imported VimSketchDataset and NAME_TO_WIDTH!")
except ImportError as e:
    print(f"Failed to import VimSketchDataset or NAME_TO_WIDTH: {e}")

try:
    from audioldm.vocaldm_utils import load_audioldm_model, setup_qvim_and_adapter
    print("Successfully imported from vocaldm_utils!")
except ImportError as e:
    print(f"Failed to import from vocaldm_utils: {e}")

# Attempt to run a very stripped down version of train_vocaldm.py
try:
    print("\nAttempting to access QVIM model directory:")
    qvim_checkpoint = "audioldm/qvim/models_vimsketch_longer/dulcet-leaf-31/best-mrr-checkpoint.ckpt"
    if os.path.exists(qvim_checkpoint):
        print(f"QVIM checkpoint found at: {qvim_checkpoint}")
    else:
        print(f"QVIM checkpoint NOT found at: {qvim_checkpoint}")
        
    # Check if this path is absolute or relative
    if os.path.isabs(qvim_checkpoint):
        print("Path is absolute")
    else:
        print("Path is relative, absolute path would be:", os.path.abspath(qvim_checkpoint))
        
except Exception as e:
    print(f"Error checking QVIM checkpoint: {e}")