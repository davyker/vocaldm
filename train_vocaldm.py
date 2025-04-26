#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import warnings

# Silence warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader, Dataset
import wandb
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import sys
import gc
import matplotlib.pyplot as plt
from matplotlib import cm
import atexit
import random
from torchviz import make_dot

# Import from AudioLDM modules
from audioldm.latent_diffusion.ddpm import DDPM
from audioldm.utils import save_wave
from audioldm.variational_autoencoder.autoencoder import AutoencoderKL
from audioldm.qvim_adapter import extract_qvim_embedding, prepare_vocim_conditioning
from audioldm.qvim.src.qvim_mn_baseline.dataset import VimSketchDataset
from audioldm.qvim.src.qvim_mn_baseline.utils import NAME_TO_WIDTH
from audioldm.qvim.src.qvim_mn_baseline.download import download_vimsketch_dataset

# Debug patch for torch.autograd.grad to find tensor gradient issues
try:
    import audioldm.latent_diffusion.util as ldm_util
    
    # Get the debug flag from command line arguments - default to False
    import sys
    debug_autograd = "--debug_autograd" in sys.argv
    
    # Only apply patches if debug mode is enabled
    if debug_autograd:
        # Also patch torch.autograd.grad directly to identify exactly which tensors cause the error
        import torch.autograd
        original_grad = torch.autograd.grad
        
        def debug_grad_wrapper(*args, **kwargs):
            print("\n-------- Entering patched torch.autograd.grad --------")
            try:
                # Extract the tensors being differentiated
                if len(args) >= 2:
                    outputs, inputs = args[0], args[1]
                    if isinstance(outputs, torch.Tensor):
                        print(f"Output tensor: requires_grad={outputs.requires_grad}, shape={outputs.shape}")
                    if isinstance(inputs, (list, tuple)):
                        print(f"Number of input tensors: {len(inputs)}")
                        for i, inp in enumerate(inputs):
                            if isinstance(inp, torch.Tensor):
                                print(f"Input tensor {i}: requires_grad={inp.requires_grad}, shape={inp.shape}")
                    elif isinstance(inputs, torch.Tensor):
                        print(f"Single input tensor: requires_grad={inputs.requires_grad}, shape={inputs.shape}")
                # Call the original function
                return original_grad(*args, **kwargs)
            except RuntimeError as e:
                # Special handling for the requires_grad error
                if "One of the differentiated Tensors does not require grad" in str(e):
                    print("\n!!! CAUGHT THE REQUIRES_GRAD ERROR !!!")
                    # Extract the tensors being differentiated
                    if len(args) >= 2:
                        outputs, inputs = args[0], args[1]
                        
                        print("\nDetailed tensor analysis:")
                        # Check outputs
                        if isinstance(outputs, torch.Tensor):
                            print(f"Output tensor: requires_grad={outputs.requires_grad}, shape={outputs.shape}")
                            
                        # Check inputs thoroughly - this is likely where the error is
                        if isinstance(inputs, (list, tuple)):
                            print(f"Input tensors: {len(inputs)}")
                            for i, inp in enumerate(inputs):
                                if isinstance(inp, torch.Tensor):
                                    print(f"Input {i}: requires_grad={inp.requires_grad}, shape={inp.shape}, dtype={inp.dtype}")
                                    # Investigate if it's a leaf or has a grad_fn
                                    print(f"  is_leaf={inp.is_leaf}, has grad_fn={inp.grad_fn is not None}")
                                    if not inp.requires_grad:
                                        print(f"  !!! THIS TENSOR DOESN'T REQUIRE GRAD - LIKELY THE CAUSE OF THE ERROR !!!")
                        elif isinstance(inputs, torch.Tensor):
                            print(f"Single input: requires_grad={inputs.requires_grad}, shape={inputs.shape}")
                            if not inputs.requires_grad:
                                print(f"  !!! THIS TENSOR DOESN'T REQUIRE GRAD - LIKELY THE CAUSE OF THE ERROR !!!")
                raise
            finally:
                print("-------- Exiting patched grad function --------\n")
        
        # Apply the patch to torch.autograd.grad only in debug mode
        torch.autograd.grad = debug_grad_wrapper
        print("Successfully applied debug patch to torch.autograd.grad")
    else:
        print("Running without autograd debug patches (use --debug_autograd to enable)")
    
except Exception as e:
    print(f"Failed to apply debug patches: {e}")

# Import utility functions
from vocaldm_utils import load_audioldm_model_with_qvim_cond, setup_qvim_and_adapter, cleanup_resources, make_vocaldm_batch

# Enable Tensor Cores for faster training with minimal precision loss
torch.set_float32_matmul_precision('high')

class VocaLDMDataset(Dataset):
    """Dataset for training VocaLDM with vocal imitations and reference sounds"""
    
    def __init__(self, base_dataset, sample_rate=32000, audioldm_sample_rate=16000, duration=10.0, max_items=None):
        """
        Args:
            base_dataset: VimSketchDataset or similar dataset with imitations and references
            sample_rate: Target sample rate for audio processing (QVIM sample rate)
            audioldm_sample_rate: AudioLDM sample rate
            duration: Target duration in seconds
            max_items: Optional limit on dataset size for debugging/testing
        """
        self.base_dataset = base_dataset
        self.sample_rate = sample_rate
        self.audioldm_sample_rate = audioldm_sample_rate
        self.duration = duration
        self.max_items = max_items
        
        # If max_items is set, limit the dataset size
        if max_items is not None:
            self.valid_indices = list(range(min(max_items, len(base_dataset))))
        else:
            self.valid_indices = list(range(len(base_dataset)))
            
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        # Get the original item from the base dataset
        original_idx = self.valid_indices[idx]
        item = self.base_dataset[original_idx]
        
        # Extract components we need for training
        imitation = item['imitation']
        reference = item['reference']
        imitation_filename = item['imitation_filename']
        reference_filename = item['reference_filename']
        
        # Create a batch-friendly item structure
        result = {
            'imitation': imitation,
            'reference': reference,
            'imitation_filename': imitation_filename,
            'reference_filename': reference_filename
        }
        
        # Include mel_reference if it exists (using AudioLDM's original processing)
        if 'mel_reference' in item:
            result['mel_reference'] = item['mel_reference']
            
        return result

class VocaLDMModule(pl.LightningModule):
    """
    PyTorch Lightning module for training VocaLDM (AudioLDM conditioned on vocal imitations)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        # self.param_names_contain = ['film', 'emb_layers']
        self.param_names_contain = config.param_names_contain
        
        # Store current loss values for scheduler
        self.current_train_loss = float('inf')
        
        # Initialize models (QVIM, AudioLDM, and adapter)
        self.initialize_models()
        
        # Selectively freeze/unfreeze parameters
        self.freeze_model_except_adapter_and_film_layers()
        
        # Training metrics
        self.train_step_outputs = []
        self.validation_step_outputs = []
    
    @property
    def is_global_zero(self):
        """Check if this process is the global zero rank (or single process)"""
        return (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or not torch.distributed.is_initialized()
        
    def on_after_backward(self):
        """Monitor gradient magnitudes after backward pass to detect vanishing gradients"""
        # Only log every 10 steps to avoid flooding logs
        if self.global_step % 10 == 0:
            # Track adapter gradients
            adapter_grads = []
            for name, param in self.adapter.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    adapter_grads.append(grad_norm)
                    
                    # Log specific layer gradients for debugging
                    if self.global_step % 100 == 0 and self.is_global_zero and self.logger:
                        self.logger.experiment.log({f"grad/adapter_{name}": grad_norm})
            
            # Track FiLM gradients
            film_grads = []
            for name, param in self.audioldm.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if any(x in name for x in self.param_names_contain):
                        grad_norm = param.grad.norm().item()
                        film_grads.append(grad_norm)
                        
                        # Log specific layer gradients for debugging
                        if self.global_step % 100 == 0 and self.is_global_zero and self.logger:
                            self.logger.experiment.log({f"grad/film_{name}": grad_norm})
            
            # Log gradient statistics
            if self.is_global_zero and self.logger:
                if adapter_grads:
                    self.logger.experiment.log({
                        "grad/adapter_mean": np.mean(adapter_grads),
                        "grad/adapter_median": np.median(adapter_grads),
                        "grad/adapter_max": max(adapter_grads),
                        "grad/adapter_min": min(adapter_grads),
                    })
                    
                    # Flag abnormally small gradients
                    if min(adapter_grads) < 1e-5:
                        self.logger.experiment.log({"grad/adapter_vanishing": 1.0})
                
                if film_grads:
                    self.logger.experiment.log({
                        "grad/film_mean": np.mean(film_grads),
                        "grad/film_median": np.median(film_grads),
                        "grad/film_max": max(film_grads),
                        "grad/film_min": min(film_grads),
                    })
                    
                    # Flag abnormally small gradients
                    if min(film_grads) < 1e-5:
                        self.logger.experiment.log({"grad/film_vanishing": 1.0})
        
    def initialize_models(self):
        """Initialize QVIM model, AudioLDM, and adapter"""
        # Load QVIM model
        print(f"Loading QVIM model from {self.config.qvim_checkpoint}")
        self.qvim_model, self.adapter = setup_qvim_and_adapter(
            self.config.qvim_checkpoint,
            self.config.qvim_dim,
            self.config.audioldm_dim
        )
        self.qvim_model.requires_grad_(False)  # Ensure QVIM model is frozen
        
        # Load AudioLDM model
        model_source = self.config.audioldm_checkpoint or self.config.audioldm_model
        self.audioldm = load_audioldm_model_with_qvim_cond(model_source, device=self.device)
        
        # CRITICAL FIX: Use our completely training-compatible checkpoint function and class
        # The original checkpoint function is causing "One of the differentiated Tensors does not require grad" errors
        # This is a more comprehensive fix that replaces both the function and the autograd.Function class
        import audioldm.latent_diffusion.util as util
        from audioldm.latent_diffusion.util_for_training import checkpoint_full_replacement, CheckpointFunctionForTraining
        
        # Save the original checkpoint function and class
        original_checkpoint = util.checkpoint
        original_checkpoint_function = util.CheckpointFunction
        
        # Replace them with our training-compatible versions
        print("Replacing checkpoint function and CheckpointFunction class with training-compatible versions")
        util.checkpoint = checkpoint_full_replacement
        util.CheckpointFunction = CheckpointFunctionForTraining
        
        # Register a cleanup function to restore the original components when done
        def restore_checkpoint():
            util.checkpoint = original_checkpoint
            util.CheckpointFunction = original_checkpoint_function
            print("Restored original checkpoint function and class")
            
        import atexit
        atexit.register(restore_checkpoint)
        
        # Also fix the latent encoding detachment in LDM
        # This is a critical issue in ldm.py line 191: .detach() breaks gradient flow
        if args.debug_autograd:
            import types
            
            # Create a training-compatible version of get_first_stage_encoding
            def get_first_stage_encoding_with_grads(self, encoder_posterior):
                """Training-compatible version that preserves gradients"""
                if isinstance(encoder_posterior, torch.Tensor):
                    z = encoder_posterior
                elif hasattr(encoder_posterior, 'sample'):
                    z = encoder_posterior.sample()
                else:
                    raise NotImplementedError(f"Type {type(encoder_posterior)} not supported")
                # Critical difference: no .detach() here
                return self.scale_factor * z
            
            # Monkey-patch the get_first_stage_encoding method to preserve gradients
            print("Replacing get_first_stage_encoding method to preserve gradients")
            self.audioldm.get_first_stage_encoding = types.MethodType(
                get_first_stage_encoding_with_grads, 
                self.audioldm
            )
        
        # Make sure the adapter is in training mode
        self.adapter.train()
    
    def save_parameter_names(self):
        """Save all parameter names and shapes to a text file in the run directory"""
        if hasattr(self.trainer, 'run_dir'):
            params_file = os.path.join(self.trainer.run_dir, "model_parameters.txt")
            simplified_file = os.path.join(self.trainer.run_dir, "simplified_parameters.txt")
        else:
            params_file = "model_parameters.txt"
            simplified_file = "simplified_parameters.txt"
            
        # Create set for simplified parameter names and a dict to track parameter counts
        simplified_params = {}
        
        # Helper function to simplify parameter names
        def simplify_name(name):
            # Remove weight/bias suffixes
            if name.endswith(".weight") or name.endswith(".bias"):
                name = name[:-len(".weight" if name.endswith(".weight") else ".bias")]
            
            # Remove purely numeric parts
            parts = name.split(".")
            result = [part for part in parts if not part.isdigit()]
            
            return ".".join(result)
            
        # Save full parameter details
        with open(params_file, 'w') as f:
            # First save audioldm parameters
            f.write("=== AudioLDM Parameters ===\n")
            for name, param in self.audioldm.named_parameters():
                num_params = param.numel() / 1000  # Convert to thousands
                f.write(f"{name}, Shape: {param.shape}, Params: {num_params:.1f}K, Requires grad: {param.requires_grad}\n")
                
                # Add parameter count to simplified name
                simple_name = simplify_name("audioldm." + name)
                if simple_name in simplified_params:
                    simplified_params[simple_name] += param.numel()
                else:
                    simplified_params[simple_name] = param.numel()
                
            # Then save adapter parameters
            f.write("\n=== Adapter Parameters ===\n")
            for name, param in self.adapter.named_parameters():
                num_params = param.numel() / 1000  # Convert to thousands
                f.write(f"{name}, Shape: {param.shape}, Params: {num_params:.1f}K, Requires grad: {param.requires_grad}\n")
                
                # Add parameter count to simplified name
                simple_name = simplify_name("adapter." + name)
                if simple_name in simplified_params:
                    simplified_params[simple_name] += param.numel()
                else:
                    simplified_params[simple_name] = param.numel()
                
            # Finally add total counts
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            f.write(f"\nTotal Parameters: {total_params:,} ({total_params/1000:.1f}K)\n")
            f.write(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1000:.1f}K)\n")
        
        # Save simplified parameter names with their total counts
        with open(simplified_file, 'w') as f:
            f.write("=== Simplified Parameter Blocks ===\n")
            for name, count in sorted(simplified_params.items()):
                f.write(f"{name}, Params: {count/1000:.1f}K\n")
            f.write(f"\nTotal Unique Parameter Blocks: {len(simplified_params)}\n")
                
        print(f"Parameter names saved to: {params_file}")
        print(f"Simplified parameter blocks saved to: {simplified_file}")
    
    def freeze_model_except_adapter_and_film_layers(self):
        """
        Freeze all parameters except:
        1. The adapter (which needs to be trained)
        2. The FiLM conditioning layers that process embeddings
        """
        # First freeze everything
        self.audioldm.eval()  # Set AudioLDM to eval mode
        self.qvim_model.eval()  # Set QVIM model to eval mode
        self.adapter.train()  # Make sure adapter is in train mode
        
        # CRITICAL FIX: Disable ALL gradient checkpointing in the model
        # This addresses the "One of the differentiated Tensors does not require grad" error
        print("Searching for and disabling ALL gradient checkpointing in the model...")
        checkpoint_count = 0
        
        # First check the entire AudioLDM model for any use_checkpoint attributes
        for name, module in self.audioldm.named_modules():
            if hasattr(module, 'use_checkpoint'):
                if module.use_checkpoint:
                    print(f"Disabling gradient checkpointing in {name}")
                    module.use_checkpoint = False
                    checkpoint_count += 1
        
        # Specifically check the diffusion model which is the most likely to use checkpointing
        if hasattr(self.audioldm, 'model') and hasattr(self.audioldm.model, 'diffusion_model'):
            diffusion_model = self.audioldm.model.diffusion_model
            for name, module in diffusion_model.named_modules():
                if hasattr(module, 'use_checkpoint'):
                    if module.use_checkpoint:
                        print(f"Disabling gradient checkpointing in diffusion_model.{name}")
                        module.use_checkpoint = False
                        checkpoint_count += 1
        
        print(f"Disabled gradient checkpointing in {checkpoint_count} modules")
        
        for param in self.audioldm.parameters():
            param.requires_grad = False
            
        # Unfreeze FiLM conditioning layers
        film_params = []
        film_param_count = 0
        
        # Find and unfreeze parameters in the diffusion model that handle conditioning
        # AudioLDM typically has conditioning projection layers in the model
        for name, module in self.audioldm.named_modules():
            # Look for FiLM conditioning layers - typically linear projections in the time embedder
            # or parameters with 'cond' in their name
            if any(x in name for x in self.param_names_contain): # Removed 'time_embed'
                print(f"Unfreezing FiLM parameters in: {name}") if args.debug_autograd else None
                for param_name, param in module.named_parameters():
                    param.requires_grad = True
                    film_params.append(param)
                    film_param_count += param.numel()
                
                # Set this specific module to train mode even though the parent model is in eval mode
                module.train()
        
        # # Extra check: also look for 'film_emb' directly in model parameters
        # for name, param in self.audioldm.named_parameters():
        #     if any(x in name for x in ['film_emb', 'cond_emb']):
        #         if not param.requires_grad:
        #             print(f"Found and unfreezing additional FiLM parameter: {name}")
        #             param.requires_grad = True
        #             film_params.append(param)
        #             film_param_count += param.numel()
        
        print(f"Unfroze {len(film_params)} FiLM parameter tensors")
        print(f"Total FiLM parameters ufrozen: {film_param_count}")
        if args.debug_autograd:
            if film_params:
                print(f"First few parameters: {film_params[:2]}")
            else:
                print("No FiLM parameters found. You may need to adjust the search criteria.")
        
        # Ensure the adapter is trainable
        for param in self.adapter.parameters():
            param.requires_grad = True
        
        adapter_param_count = sum(p.numel() for p in self.adapter.parameters() if p.requires_grad)
        print(f"Adapter has {adapter_param_count} trainable parameters")
        
        # Verify trainable parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")
        
        # Count modules in train vs eval mode
        train_count = 0
        eval_count = 0
        for module in self.modules():
            if module.training:
                train_count += 1
            else:
                eval_count += 1
        print(f"Modules in train mode: {train_count}")
        print(f"Modules in eval mode: {eval_count}")
    
    def forward(self, imitation, reference=None):
        """
        Args:
            imitation: Vocal imitation audio tensor [B, T]
            reference: Optional reference sound [B, T]
            
        Returns:
            Generated audio tensor
        """
        # During inference, we use imitation, but during training we'll use reference
        # This keeps forward() behavior the same for inference
        qvim_embedding = extract_qvim_embedding(imitation, self.qvim_model, audio_type="imitation")
        
        # Ensure AudioLDM is configured for bypass mode
        from audioldm.pipeline import set_cond_qvim
        self.audioldm = set_cond_qvim(self.audioldm)
        
        # Get latent dimensions for sampling
        shape = (self.audioldm.channels, self.audioldm.latent_t_size, self.audioldm.latent_f_size)
        batch_size = imitation.shape[0]
        
        # Adapt the embeddings using our trainable adapter
        # This transforms from QVIM embedding dimension to AudioLDM embedding dimension
        adapted_embeddings = self.adapter(qvim_embedding)
        
        # Generate unconditional embeddings for classifier-free guidance if needed
        if self.config.guidance_scale > 1.0:
            unconditional_embedding = self.adapter.get_unconditional_embedding(
                batch_size=batch_size, 
                device=self.device
            )
        else:
            unconditional_embedding = None
        
        # Use the regular DDIM sampler (which now has no @torch.no_grad decorators)
        from audioldm.latent_diffusion.ddim import DDIMSampler
        sampler = DDIMSampler(self.audioldm)
        sampler.make_schedule(ddim_num_steps=self.config.ddim_steps, ddim_eta=1.0, verbose=False)
        
        # Run sampling WITH gradient flow (no torch.no_grad())
        samples, _ = sampler.sample(
            S=self.config.ddim_steps,
            batch_size=batch_size,
            shape=shape,
            conditioning=adapted_embeddings,  # Direct tensor input
            unconditional_guidance_scale=self.config.guidance_scale,
            unconditional_conditioning=unconditional_embedding,
            verbose=False
        )
        
        # Check for extreme values and clip if needed
        if torch.max(torch.abs(samples)) > 1e2:
            samples = torch.clip(samples, min=-10, max=10)
        
        # Decode the latent space to get waveform
        mel = self.audioldm.decode_first_stage(samples)
        # Show shape of mel
        print(f"Decoded mel shape: {mel.shape}")
        
        # Extract waveform
        waveform = self.audioldm.mel_spectrogram_to_waveform(mel)
        
        # Convert to torch tensor
        waveform_tensor = torch.tensor(waveform, device=self.device)
        
        return waveform_tensor
    
    def training_step(self, batch, batch_idx):
        """
        Training step for the diffusion model with QVIM embeddings
        """
        # Get imitation and reference audio
        imitation = batch['imitation']
        reference = batch['reference']
        
        # Set modules to appropriate modes
        self.adapter.train()
        for param in self.adapter.parameters():
            param.requires_grad = True
            
        # Get QVIM embeddings for reference audio - no grad needed for feature extraction
        with torch.no_grad():
            qvim_embedding = extract_qvim_embedding(reference, self.qvim_model, audio_type="sound")  # Use reference audio with sound type
            qvim_embedding = qvim_embedding.detach()  # Make sure it's detached
            
            # Use pre-processed mel_reference - we need to ensure it exists in the dataset
            mel_reference = batch['mel_reference']
                
            # Encode the reference spectrogram - no grad needed for VAE encoding
            z_reference = self.audioldm.encode_first_stage(mel_reference)
            z_reference = self.audioldm.get_first_stage_encoding(z_reference).detach()
        
        # Sample noise and timestep
        b, *_ = z_reference.shape
        t = torch.randint(0, self.audioldm.num_timesteps, (b,), device=self.device).long()
        noise = torch.randn_like(z_reference)
        
        # Forward diffusion to get noisy latent - MUST have gradient tracking for backprop
        z_noisy = self.audioldm.q_sample(z_reference, t, noise=noise)
        
        # Important: For the parts that need gradients, run them outside torch.no_grad()
        # This is the adapter part that transforms QVIM embeddings
        adapted_embedding = self.adapter(qvim_embedding)
        
        # For training both the adapter and FiLM conditioning layers, 
        # we need to pass the adapted embedding to the diffusion model
        # with gradients enabled
        
        # Create conditioning dictionary for AudioLDM
        # The c_film key is what AudioLDM expects for FiLM conditioning
        cond = {"c_film": [adapted_embedding]}
        
        # Make sure adapted_embedding has gradients enabled
        if not adapted_embedding.requires_grad:
            print("Warning: adapted_embedding doesn't require grad - enabling it")
            adapted_embedding.requires_grad_(True)
            
        # # Make sure the FiLM layers are in training mode
        # for name, module in self.audioldm.named_modules():
        #     if any(x in name for x in ['film']): # Removed 'time_embed'
        #         module.train()
            
        # Get model prediction with gradients enabled for the FiLM layers
        try:
            # Register hooks to debug gradient issues - only in debug mode
            if self.global_step == 0 and batch_idx == 0 and hasattr(self.config, 'debug_autograd') and self.config.debug_autograd:
                # Create a hook to debug tensor requires_grad
                def debug_grad_hook(module, grad_input, grad_output):
                    # Check for Nones or tensors that don't require grad
                    if grad_input is not None:
                        for i, g in enumerate(grad_input):
                            if g is not None and not g.requires_grad:
                                print(f"Module {module.__class__.__name__} received input {i} that doesn't require_grad")
                    
                    # Check output grads
                    if grad_output is not None:
                        for i, g in enumerate(grad_output):
                            if g is not None and not g.requires_grad:
                                print(f"Module {module.__class__.__name__} produced output {i} that doesn't require_grad")
                
                # Register hook on key modules
                hooks = []
                for name, module in self.audioldm.named_modules():
                    if any(x in name for x in self.param_names_contain): # Removed 'time_embed'
                        hooks.append(module.register_backward_hook(debug_grad_hook))
                
                # Print debug info about input tensors
                print(f"z_noisy requires_grad: {z_noisy.requires_grad}")
                print(f"t requires_grad: {t.requires_grad}")
                print(f"adapted_embedding requires_grad: {adapted_embedding.requires_grad}")
                
                # Get all parameters that require gradients
                trainable_param_names = [name for name, param in self.named_parameters() if param.requires_grad]
                print(f"Trainable parameters: {trainable_param_names}")
                
                # CRITICAL FIX: Override torch.autograd.grad to filter out non-trainable parameters
                # This is the most likely solution to the "One of the differentiated Tensors does not require grad" error
                original_autograd_grad = torch.autograd.grad
                
                def filtered_grad(*args, **kwargs):
                    """
                    Custom grad function that filters out tensors that don't require gradients.
                    This is specifically designed to fix the issue with some UNet parameters
                    being included in grad computation when they shouldn't be.
                    """
                    if len(args) > 1 and isinstance(args[1], (list, tuple)):
                        outputs, inputs = args[0], args[1]
                        
                        # Only include trainable inputs for gradient computation
                        trainable_inputs = []
                        for inp in inputs:
                            if isinstance(inp, torch.Tensor) and inp.requires_grad:
                                trainable_inputs.append(inp)
                        
                        # Call original grad with filtered inputs
                        print(f"Filtered inputs from {len(inputs)} to {len(trainable_inputs)} tensors")
                        # Modify args by replacing the inputs with trainable_inputs
                        new_args = (outputs, trainable_inputs) + args[2:]
                        return original_autograd_grad(*new_args, **kwargs)
                    else:
                        # If the format is different, use the original function
                        return original_autograd_grad(*args, **kwargs)
                
                # Apply the override - this is a critical fix that should solve the error
                torch.autograd.grad = filtered_grad
            
            # For training, it's cleaner to disable EMA completely
            # Save the original EMA state
            original_use_ema = self.audioldm.use_ema
            self.audioldm.use_ema = False
            
            try:
                # Run the diffusion model with the adapter output as conditioning
                model_output = self.audioldm.apply_model(z_noisy, t, cond)
            finally:
                # Restore the original EMA state
                self.audioldm.use_ema = original_use_ema
            
            # Calculate loss based on model parameterization (typically predicting noise)
            if self.audioldm.parameterization == "eps":
                # Important: Using the noise directly without .detach() to see if this helps
                # While typically we'd detach the target, not doing so might help with gradient flow
                target = noise
            elif self.audioldm.parameterization == "x0":
                target = z_reference
            else:
                raise ValueError(f"Unknown parameterization: {self.audioldm.parameterization}")
                
            # Check if model_output requires gradients - it MUST for proper backpropagation
            # Just log a warning if it doesn't, but don't modify the tensor as that would break the computational graph
            if not model_output.requires_grad:
                print("WARNING: model_output doesn't require grad - this indicates an issue with gradient flow!")
                print("The problem might be in apply_model function or earlier in the pipeline.")
            
            # Add additional debugging to check tensor states just before error point - only in debug mode
            if self.global_step == 0 and batch_idx == 0 and hasattr(self.config, 'debug_autograd') and self.config.debug_autograd:
                print(f"model_output shape: {model_output.shape}, requires_grad: {model_output.requires_grad}")
                print(f"target shape: {target.shape}, requires_grad: {target.requires_grad}")
                
                # Check the conditioning structure for better debugging
                print(f"Conditioning structure (cond): {type(cond)}")
                for key, value in cond.items():
                    if isinstance(value, list):
                        print(f"  Key: {key}, Value type: List with {len(value)} items")
                        for i, item in enumerate(value):
                            if isinstance(item, torch.Tensor):
                                print(f"    Item {i}: Tensor shape={item.shape}, requires_grad={item.requires_grad}")
                    else:
                        print(f"  Key: {key}, Value type: {type(value)}")
                
                # Examine model properties
                print(f"Model conditioning key: {self.audioldm.model.conditioning_key}")
                print(f"Model parameterization: {self.audioldm.parameterization}")
                
                # Check what parameters influence model_output
                all_params = dict(self.named_parameters())
                try:
                    from torch.autograd import grad
                    # Test gradient flow from model_output to first pixel only to check connectivity
                    first_pixel = model_output[0, 0, 0, 0]
                    
                    # Get gradients for all parameters that require_grad=True
                    for name, param in self.named_parameters():
                        if param.requires_grad:
                            try:
                                # Try computing gradients one parameter at a time to find problematic ones
                                g = grad(first_pixel, param, retain_graph=True, allow_unused=True)[0]
                                if g is None:
                                    print(f"Warning: Parameter {name} has no gradient connection to output")
                                else:
                                    print(f"Parameter {name} is properly connected to computation graph")
                            except Exception as e:
                                print(f"Error computing gradient for {name}: {e}")
                except Exception as e:
                    print(f"Error during gradient inspection: {e}")
            
            # This is the standard diffusion training loss, but we'll be careful about it
            # We'll manually compute MSE with explicit gradient tracking
            squared_diff = (model_output - target)**2
            diffusion_loss = squared_diff.mean()
            
            # Verify the loss has gradients properly attached
            if self.global_step == 0 and batch_idx == 0:
                print(f"diffusion_loss requires_grad: {diffusion_loss.requires_grad}")
                print(f"Checking gradient paths:")
                
                # Check if every component in the gradient calculation has requires_grad correctly set
                components_check = {
                    'squared_diff': squared_diff.requires_grad,
                    'model_output': model_output.requires_grad,
                    'target': target.requires_grad
                }
                print(f"Component requires_grad check: {components_check}")
            
            # The diffusion loss is our main loss for training both components
            loss = diffusion_loss
            
            # Visualize autograd graph at first step of training - only in debug mode
            if self.global_step == 0 and batch_idx == 0 and hasattr(self.config, 'debug_autograd') and self.config.save_autograd_graph:
                print("Visualizing autograd computation graph...")
                os.makedirs("debug", exist_ok=True)
                # Visualize from loss tensor back to all parameters
                graph = make_dot(loss, params=dict(self.named_parameters()))
                graph.render("debug/autograd_graph", format="png")
                print("Autograd graph saved to debug/autograd_graph.png")
                
                # Also visualize just the FiLM conditioning path
                film_graph = make_dot(model_output, params=dict(self.named_parameters()))
                film_graph.render("debug/film_conditioning_graph", format="png")
                print("FiLM conditioning graph saved to debug/film_conditioning_graph.png")
            
            # Keep track of component losses for logging
            mse_loss = diffusion_loss
            cosine_loss = torch.tensor(0.0, device=self.device)  # Placeholder
            
        except RuntimeError as e:
            # If we hit the gradient error, fall back to a simplified approach
            # that only trains the adapter without using the diffusion model
            print(f"Warning: Diffusion training failed with error: {e}")
            print("Falling back to adapter-only training")
            
            # Get batch size from input shape
            actual_batch_size = adapted_embedding.shape[0]
            
            # Structural preservation - make sure the adapter maintains relative similarities
            if actual_batch_size > 1:
                # Calculate similarity matrices
                qvim_similarity = F.cosine_similarity(
                    qvim_embedding.unsqueeze(1), 
                    qvim_embedding.unsqueeze(0), 
                    dim=-1
                )
                
                adapted_similarity = F.cosine_similarity(
                    adapted_embedding.unsqueeze(1), 
                    adapted_embedding.unsqueeze(0), 
                    dim=-1
                )
                
                # Structural loss
                structure_loss = F.mse_loss(adapted_similarity, qvim_similarity)
            else:
                structure_loss = torch.tensor(0.0, device=self.device)
            
            # Format loss components
            norms = torch.norm(adapted_embedding, p=2, dim=-1)
            unit_norm_loss = F.mse_loss(norms, torch.ones_like(norms))
            distribution_loss = torch.mean(torch.abs(adapted_embedding.mean(dim=0)))
            
            # Define fallback loss
            loss = 10.0 * structure_loss + unit_norm_loss + 0.1 * distribution_loss
            
            # For logging purposes
            mse_loss = unit_norm_loss
            cosine_loss = structure_loss
        
        # Store current loss value
        self.current_train_loss = loss.item()
        
        # Standard Lightning logging for progress bar
        self.log('train/loss', loss, prog_bar=True, batch_size=imitation.shape[0])
        self.log('train/mse_loss', mse_loss, batch_size=imitation.shape[0])
        self.log('train/cosine_loss', cosine_loss, batch_size=imitation.shape[0])
        
        # Log learning rates
        # Get optimizer
        optimizer = self.optimizers()
        
        # # Show initial parameter group information on first step
        # if self.global_step == 0:
        #     print("Optimizer parameter groups:")
        #     for i, group in enumerate(optimizer.param_groups):
        #         print(f"  Group {i}: lr={group['lr']}, weight_decay={group['weight_decay']}, {len(group['params'])} parameters")
        
        # Get current learning rates and weight decay values
        adapter_lr = optimizer.param_groups[0]['lr']
        film_lr = optimizer.param_groups[1]['lr']
        adapter_wd = optimizer.param_groups[0]['weight_decay']
        film_wd = optimizer.param_groups[1]['weight_decay']
        
        # Standard Lightning logging for progress bar
        self.log('train/adapter_lr', adapter_lr, prog_bar=True, batch_size=imitation.shape[0])
        self.log('train/film_lr', film_lr, prog_bar=True, batch_size=imitation.shape[0])
        
        # Direct WandB logging for reliable charts - only on rank 0
        if self.is_global_zero and self.logger and hasattr(self.logger, 'experiment'):
            self.logger.experiment.log({
                'train/loss': loss.item(),
                'train/mse_loss': mse_loss.item(),
                'train/cosine_loss': cosine_loss.item(),
                'train/adapter_lr': adapter_lr,
                'train/film_lr': film_lr,
                'train/adapter_weight_decay': adapter_wd,
                'train/film_weight_decay': film_wd,
                'step': self.global_step,
                'epoch': self.current_epoch
            })
        
        # Append to outputs for epoch-end processing
        self.train_step_outputs.append(loss.detach())
            
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        # Get imitation and reference audio
        imitation = batch['imitation']
        reference = batch['reference']
        
        # Set adapter to eval mode for validation
        self.adapter.eval()
        
        with torch.no_grad():
            # Get QVIM embeddings for reference audio
            qvim_embedding = extract_qvim_embedding(reference, self.qvim_model, audio_type="sound")
            
            # Adapt the embeddings using our trainable adapter (in eval mode for validation)
            adapted_embedding = self.adapter(qvim_embedding)
            
            # Use pre-processed mel_reference - we need to ensure it exists in the dataset
            mel_reference = batch['mel_reference']
                
            # Encode the reference spectrogram
            z_reference = self.audioldm.encode_first_stage(mel_reference)
            z_reference = self.audioldm.get_first_stage_encoding(z_reference).detach()
            
            # Sample noise and timestep
            batch_size = z_reference.shape[0]
            t = torch.randint(0, self.audioldm.num_timesteps, (batch_size,), device=self.device).long()
            noise = torch.randn_like(z_reference)
            
            # Forward diffusion to get noisy latent
            z_noisy = self.audioldm.q_sample(z_reference, t, noise=noise)
            
            # Create conditioning for AudioLDM evaluation
            cond = {"c_film": [adapted_embedding]}
            
            # Try to run the diffusion model for evaluation
            try:
                # Get model prediction - make sure it's trainable with gradients
                model_output = self.audioldm.apply_model(z_noisy, t, cond)
                
                # Calculate loss based on model parameterization
                if self.audioldm.parameterization == "eps":
                    # Use detach in validation too for consistency
                    target = noise.detach()
                elif self.audioldm.parameterization == "x0":
                    target = z_reference.detach()
                else:
                    raise ValueError(f"Unknown parameterization: {self.audioldm.parameterization}")
                
                # Standard diffusion loss
                diffusion_loss = F.mse_loss(model_output, target)
                
                # Use diffusion loss as our main validation metric
                loss = diffusion_loss
                
                # For logging
                mse_loss = diffusion_loss
                cosine_loss = torch.tensor(0.0, device=self.device)
                
            except RuntimeError as e:
                # Fall back to the same approach as in training
                print(f"Warning: Validation diffusion failed with error: {e}")
                print("Falling back to adapter-only validation")
                
                # Get actual batch size from input shape
                actual_batch_size = adapted_embedding.shape[0]
                
                # Structural preservation
                if actual_batch_size > 1:
                    # Calculate similarity matrices
                    qvim_similarity = F.cosine_similarity(
                        qvim_embedding.unsqueeze(1), 
                        qvim_embedding.unsqueeze(0), 
                        dim=-1
                    )
                    
                    adapted_similarity = F.cosine_similarity(
                        adapted_embedding.unsqueeze(1), 
                        adapted_embedding.unsqueeze(0), 
                        dim=-1
                    )
                    
                    # Structural loss
                    structure_loss = F.mse_loss(adapted_similarity, qvim_similarity)
                else:
                    structure_loss = torch.tensor(0.0, device=self.device)
                
                # Format metrics
                norms = torch.norm(adapted_embedding, p=2, dim=-1)
                unit_norm_loss = F.mse_loss(norms, torch.ones_like(norms))
                distribution_loss = torch.mean(torch.abs(adapted_embedding.mean(dim=0)))
                                
                # Total loss
                loss = 10.0 * structure_loss + unit_norm_loss + 0.1 * distribution_loss
                
                # For logging
                mse_loss = unit_norm_loss
                cosine_loss = structure_loss
        
        # Standard Lightning logging for progress bar and checkpointing
        self.log('val/loss', loss, prog_bar=True, batch_size=imitation.shape[0])
        self.log('val_loss', loss, batch_size=imitation.shape[0])  # Add underscore version for checkpoint filename
        self.log('val/cosine_loss', cosine_loss, batch_size=imitation.shape[0])
        self.log('val/mse_loss', mse_loss, batch_size=imitation.shape[0])
        
        # Direct WandB logging for reliable charts - only on rank 0
        if self.is_global_zero and self.logger and hasattr(self.logger, 'experiment'):
            self.logger.experiment.log({
                'val/loss': loss.item(),
                'val/mse_loss': mse_loss.item(),
                'val/cosine_loss': cosine_loss.item(),
                'step': self.global_step,
                'epoch': self.current_epoch
            })
        
        self.validation_step_outputs.append(loss.detach())
        
        # Generate audio for the 2nd batch only (to save compute)
        if batch_idx == 1 and self.current_epoch % self.config.generate_every_n_epochs == 0:
            # For generation in validation, we can use torch.no_grad() to save memory
            with torch.no_grad():
                # Use a small subset for generation
                subset_size = min(2, imitation.shape[0])
                subset_imitation = imitation[:subset_size]
                subset_reference = reference[:subset_size]
                
                # Generate audio and compare to reference 
                # forward() now uses DDIMSamplerForTraining instead of DDIMSampler
                generated_audio = self.forward(subset_imitation)
                
                # Log sample audio with spectrograms
                self.log_audio_examples(subset_imitation, subset_reference, generated_audio)
        
        return loss
    
    def on_train_epoch_end(self):
        # Calculate average loss
        avg_loss = torch.stack(self.train_step_outputs).mean()
        self.log('train/epoch_loss', avg_loss)
        self.train_step_outputs = []
        
        # Aggressively clean up memory before validation to prevent OOM
        import gc
        print("Cleaning up memory before validation...") if args.debug_autograd else None
        
        # Clear any cached tensors
        torch.cuda.empty_cache()
        
        # Run garbage collection multiple times to ensure everything is cleaned
        gc.collect()
        gc.collect()
    
    def on_validation_epoch_end(self):
        # Calculate average loss
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log('val/epoch_loss', avg_loss)
        self.validation_step_outputs = []
    
    def log_audio_examples(self, imitations, references, generated_audio):
        """Log audio examples for visualization in wandb and save to disk"""
        if not self.config.log_audio:
            print("Audio logging disabled (use --log_audio to enable)")
            return
            
        try:
            # Get the run directory from trainer
            run_dir = self.trainer.run_dir
            audio_dir = os.path.join(run_dir, "audio_logging")
            os.makedirs(audio_dir, exist_ok=True)
            
            # Save audio files to disk - save at AudioLDM's sample rate
            
            for i in range(min(2, len(imitations))):
                # Convert to numpy arrays
                if isinstance(imitations[i], torch.Tensor):
                    imitation_audio = imitations[i].detach().cpu().numpy()
                else:
                    imitation_audio = imitations[i]
                    
                if isinstance(references[i], torch.Tensor):
                    reference_audio = references[i].detach().cpu().numpy()
                else:
                    reference_audio = references[i]
                    
                if isinstance(generated_audio[i], torch.Tensor):
                    generated_audio_np = generated_audio[i].detach().cpu().numpy()
                else:
                    generated_audio_np = generated_audio[i]
                
                # Import soundfile for saving audio (same as AudioLDM)
                import soundfile as sf
                
                # Format filenames
                epoch_str = f"epoch_{self.current_epoch:03d}"
                imitation_path = os.path.join(audio_dir, f"{epoch_str}_sample{i}_imitation.wav")
                reference_path = os.path.join(audio_dir, f"{epoch_str}_sample{i}_reference.wav")
                generated_path = os.path.join(audio_dir, f"{epoch_str}_sample{i}_generated.wav")
                
                # Print diagnostic info before processing
                max_abs_val = np.max(np.abs(generated_audio_np))
                print(f"Generated audio max abs value: {max_abs_val:.4f}")
                
                # AudioLDM approach: Scale to int16 range or clip to [-1,1] and use soundfile
                
                # Handle dimensions for generated audio (make sure we have the right shape)
                if len(generated_audio_np.shape) >= 2:
                    # Extract the actual waveform from [channel, time] format
                    waveform = generated_audio_np[0]
                else:
                    waveform = generated_audio_np
                
                # Exactly match AudioLDM's approach: Convert directly to int16 without clipping
                
                # Convert to int16 - EXACTLY like in vocoder_infer in hifigan/utilities.py
                waveform_int16 = (waveform * 32768).astype(np.int16)
                # Use soundfile with explicitly specifying format='WAV'
                sf.write(generated_path, waveform_int16, self.config.audioldm_sample_rate, format='WAV')
                
                # Handle imitation audio with identical approach
                if len(imitation_audio.shape) >= 2:
                    imitation_waveform = imitation_audio[0]
                else:
                    imitation_waveform = imitation_audio
                
                imitation_int16 = (imitation_waveform * 32768).astype(np.int16)
                sf.write(imitation_path, imitation_int16, self.config.sample_rate, format='WAV')
                
                # Handle reference audio with identical approach
                if len(reference_audio.shape) >= 2:
                    reference_waveform = reference_audio[0]
                else:
                    reference_waveform = reference_audio
                
                reference_int16 = (reference_waveform * 32768).astype(np.int16)
                sf.write(reference_path, reference_int16, self.config.sample_rate, format='WAV')
                
                print(f"Saved audio files for epoch {self.current_epoch}, sample {i} to {audio_dir}")
                
                # Log to wandb if available - only on rank 0
                if self.is_global_zero and self.logger and hasattr(self.logger, 'experiment'):
                    try:
                        self.logger.experiment.log({
                            f"audio/epoch_{self.current_epoch}_sample_{i}_imitation": wandb.Audio(
                                imitation_path, 
                                caption=f"Epoch {self.current_epoch} - Vocal Imitation Input {i}"
                            ),
                            f"audio/epoch_{self.current_epoch}_sample_{i}_reference": wandb.Audio(
                                reference_path,
                                caption=f"Epoch {self.current_epoch} - Reference Sound {i}"
                            ),
                            f"audio/epoch_{self.current_epoch}_sample_{i}_generated": wandb.Audio(
                                generated_path,
                                caption=f"Epoch {self.current_epoch} - Generated Sound {i}"
                            )
                        })
                    except Exception as e:
                        print(f"Error logging audio to wandb: {e}")
            
            # Clean up memory after logging
            torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error in audio logging: {e}")
            import traceback
            traceback.print_exc()
            # Continue training even if logging fails
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        # Create parameter groups
        adapter_params = list(self.adapter.parameters())
        
        # Find FiLM conditioning parameters to train
        film_params = []
        for name, param in self.audioldm.named_parameters():
            if param.requires_grad and any(x in name for x in self.param_names_contain): # Removed 'time_embed'
                film_params.append(param)
        
        # Configure optimizer with two parameter groups, each with its own weight decay
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": adapter_params, 
                    "lr": self.config.adapter_lr,
                    "weight_decay": self.config.adapter_weight_decay  # Higher weight decay for new components
                },
                {
                    "params": film_params, 
                    "lr": self.config.film_lr,
                    "weight_decay": self.config.film_weight_decay  # Lower weight decay for pre-trained components
                }
            ],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Configure learning rate scheduler
        if self.config.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.lr_rampdown_epochs,  # Use separate rampdown period
                eta_min=self.config.min_lr
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",  # Changed back to epoch updates for more stability
                    "frequency": 1
                }
            }
        elif self.config.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                min_lr=self.config.min_lr
            )
            
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        else:
            return optimizer
            
    def configure_ddp(self, model, device_ids):
        """Configure Distributed Data Parallel (DDP) for multi-GPU training"""
        from pytorch_lightning.plugins.environments.lightning_environment import LightningEnvironment
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        # Set up DDP with find_unused_parameters=True to avoid DDP errors
        # This is needed because not all model parameters are used in every forward pass
        ddp = DDP(
            model,
            device_ids=device_ids,
            find_unused_parameters=True
        )
        return ddp

def save_vocaldm_checkpoint(model, path, val_loss=None, epoch=None, global_step=None):
    """
    Save a complete VocaLDM checkpoint with both adapter and FiLM parameters
    
    Args:
        model: The VocaLDM model to save
        path: Path where to save the checkpoint
        val_loss: Optional validation loss to include in the checkpoint
        epoch: Optional epoch number to include in the checkpoint
        global_step: Optional global step to include in the checkpoint
    """
    # Create a dictionary with both adapter and FiLM parameters
    save_dict = {
        'adapter': model.adapter.state_dict(),
        'state_dict': {
            # Only include parameters that require gradients
            k: v for k, v in model.state_dict().items() 
            if k.startswith('adapter.') or 
               any(x in k for x in ['film']) # Removed 'time_embed'
        }
    }
    
    # Add optional metadata
    if val_loss is not None:
        save_dict['val_loss'] = val_loss
    if epoch is not None:
        save_dict['epoch'] = epoch
    if global_step is not None:
        save_dict['global_step'] = global_step
        
    # Save the dictionary
    torch.save(save_dict, path)
    
    return path

def is_global_zero():
    """Helper function to check if this process is the global zero rank (or single process)"""
    return (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or not torch.distributed.is_initialized()

def train_vocaldm(args):
    """Main training function"""
    # Set random seed for reproducibility
    if args.seed is not None:
        pl.seed_everything(args.seed)
        
    # Enable PyTorch anomaly detection if requested
    if args.debug_autograd:
        print("Enabling PyTorch autograd anomaly detection")
        torch.autograd.set_detect_anomaly(True)
        
    # Setup CUDA error handling
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Use first GPU
        # Enable CUDA error handling to catch errors early
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Initialize wandb logger only on rank 0 for multi-GPU training
    wandb_logger = WandbLogger(
        project=args.project,
        name=args.run_name,
        config=args,
        log_model=args.wandb_log_model,
        save_dir=args.checkpoint_dir
    ) if is_global_zero() else None
    
    # Define run_id and run_dir once to use throughout the function
    if is_global_zero() and wandb_logger and wandb_logger.experiment:
        run_id = wandb_logger.experiment.name
    else:
        run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.checkpoint_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Download dataset if necessary
    try:
        download_vimsketch_dataset(args.dataset_path)
        print(f"Dataset directory ready at {args.dataset_path}")
    except Exception as e:
        print(f"Failed to download or prepare dataset: {e}")
        print(f"Check if the directory {args.dataset_path} exists and is accessible.")
        raise
    
    # Load VimSketch dataset
    try:
        dataset_path = os.path.join(args.dataset_path, 'Vim_Sketch_Dataset')
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
            
        full_ds = VimSketchDataset(
            dataset_path,
            sample_rate=args.sample_rate,
            audioldm_sample_rate=args.audioldm_sample_rate,
            duration=args.duration,
            use_original_audioldm_mel=True  # Use AudioLDM's original processing
        )
        print(f"Successfully loaded dataset with {len(full_ds)} samples")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Make sure the dataset was downloaded correctly and has the expected structure.")
        raise
    
    # Create train/validation split
    dataset_size = len(full_ds)
    val_size = int(dataset_size * args.val_split)
    train_size = dataset_size - val_size
    
    # Use random_split for better generalization
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducible split
    )
    
    # Create our wrapper datasets
    train_dataset = VocaLDMDataset(
        train_ds,
        sample_rate=args.sample_rate,
        audioldm_sample_rate=args.audioldm_sample_rate,
        duration=args.duration,
        max_items=args.max_items  # For debugging/testing
    )
    
    val_dataset = VocaLDMDataset(
        val_ds,
        sample_rate=args.sample_rate,
        audioldm_sample_rate=args.audioldm_sample_rate,
        duration=args.duration,
        max_items=args.max_items // 5 if args.max_items else None  # For debugging/testing
    )
    
    print(f"Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples")
    
    # Create data loaders with memory optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,  # Keep workers alive between batches
        prefetch_factor=2 if args.num_workers > 0 else None  # Prefetch next batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    # Initialize model
    model = VocaLDMModule(args)
    
    # Configure callbacks
    callbacks = [LearningRateMonitor(logging_interval='step')]
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        patience=args.patience,
        verbose=True,
        mode='min'
    )
    callbacks.append(early_stop_callback)
    
    # Add guidance scale scheduler if requested
    if args.use_guidance_scale_scheduler:
        guidance_scheduler = GuidanceScaleScheduler(
            initial_scale=args.initial_guidance_scale,
            target_scale=args.guidance_scale,
            warmup_percent=args.guidance_warmup_percent,
            rampup_percent=args.guidance_rampup_percent
        )
        callbacks.append(guidance_scheduler)
        print(f"Using guidance scale scheduler: {args.initial_guidance_scale} → {args.guidance_scale}")
        print(f"  Warmup: {args.guidance_warmup_percent*100}% of training, Rampup: {args.guidance_rampup_percent*100}% of training")
    
    # Model checkpointing
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        # Create a custom callback to save the adapter when validation improves
        class AdapterCheckpointCallback(pl.Callback):
            def __init__(self, checkpoint_dir):
                super().__init__()
                self.checkpoint_dir = checkpoint_dir
                self.best_val_loss = float('inf')
                
            def on_validation_end(self, trainer, pl_module):
                # Check if validation loss improved
                current_val_loss = trainer.callback_metrics.get('val/loss')
                if current_val_loss is not None and current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    
                    # Save the adapter and FiLM parameters in the run-specific directory
                    checkpoint_path = os.path.join(
                        trainer.run_dir, 
                        f"vocaldm_checkpoint_val_loss_{current_val_loss:.4f}.pt"
                    )
                    
                    # Use our helper function to save the checkpoint
                    save_vocaldm_checkpoint(
                        pl_module, 
                        checkpoint_path,
                        val_loss=current_val_loss,
                        epoch=trainer.current_epoch,
                        global_step=trainer.global_step
                    )
                    
                    print(f"\nSaved improved adapter and FiLM parameters to {checkpoint_path}")
                    
                    # Also upload to wandb if available
                    if trainer.logger and hasattr(trainer.logger, 'experiment'):
                        try:
                            trainer.logger.experiment.log_artifact(
                                checkpoint_path,
                                name=f"vocaldm_checkpoint_val_loss_{current_val_loss:.4f}",
                                type="model"
                            )
                        except Exception as e:
                            print(f"Failed to log checkpoint to wandb: {e}")
        
        # Custom checkpoint callback to save only adapter and trainable parameters
        class AdapterCheckpoint(ModelCheckpoint):
            def _save_model(self, trainer, filepath):
                # Get the model from the trainer
                model = trainer.lightning_module
                
                # Use our helper function to save the checkpoint
                save_vocaldm_checkpoint(
                    model, 
                    filepath,
                    val_loss=trainer.callback_metrics.get('val/loss'),
                    epoch=trainer.current_epoch,
                    global_step=trainer.global_step
                )
                
                # For display, add "val_loss=" to make it clearer what the number means
                val_loss = trainer.callback_metrics.get('val/loss', 0)
                display_path = filepath.replace(
                    f'-{trainer.current_epoch:02d}-{val_loss:.4f}',
                    f'-{trainer.current_epoch:02d}-val_loss={val_loss:.4f}'
                )
                print(f"Saved adapter and trainable parameters to {display_path}")
        
        print(f"Saving checkpoints to run-specific directory: {run_dir}")
        
        # Best model checkpoint using our custom class (save fewer checkpoints to reduce memory usage)
        checkpoint_callback = AdapterCheckpoint(
            dirpath=run_dir,  # Use run-specific directory
            filename='vocaldm-{epoch:02d}-{val_loss:.4f}',
            save_top_k=1,  # Reduced from 3 to 1 to save memory - only keep the very best model
            verbose=True,
            monitor='val/loss',
            mode='min',
            save_last=False  # Don't save last model to further reduce memory usage
        )
        callbacks.append(checkpoint_callback)
        
        # Commenting out last checkpoint to save memory - we'll just keep the best one
        # last_checkpoint_callback = AdapterCheckpoint(
        #     dirpath=args.checkpoint_dir,
        #     filename='vocaldm-last',
        #     save_last=True,
        #     verbose=True
        # )
        # callbacks.append(last_checkpoint_callback)
        
        # Add our custom adapter checkpoint callback
        adapter_callback = AdapterCheckpointCallback(args.checkpoint_dir)
        callbacks.append(adapter_callback)
    
    # Initialize trainer with memory optimizations
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator='auto',
        devices=args.num_gpus if torch.cuda.is_available() else None,  # 'all' or specific number of GPUs
        strategy='ddp',  # Explicitly use ddp for multi-GPU
        precision=32,  # Use full precision to avoid type mismatches
        log_every_n_steps=10,
        val_check_interval=args.val_check_interval,
        accumulate_grad_batches=args.gradient_accumulation_steps,  # Gradient accumulation for memory optimization
        gradient_clip_val=args.gradient_clip_val,  # Add gradient clipping to stabilize training
        inference_mode=False,  # Needed to avoid inference mode issues with gradient flow
        num_sanity_val_steps=-1,  # Run full validation at start for baseline metrics
        enable_checkpointing=True,  # Ensure we save the best model based on validation loss
        enable_progress_bar=True  # Show detailed progress with validation metrics
    )
    
    # Store run_dir on trainer for model access
    trainer.run_dir = run_dir
    
    # Start training with exception handling
    try:
        # Set run_dir on model to ensure parameter list is saved in right location
        model.trainer = trainer

        # Save parameter names before starting training
        model.save_parameter_names()
        
        # Run initial validation to get baseline metrics
        print("\n===== Running initial validation for baseline metrics =====")
        initial_results = trainer.validate(model, val_loader)
        
        # Log initial metrics to WandB explicitly - only on rank 0
        if is_global_zero() and wandb_logger and hasattr(wandb_logger, 'experiment'):
            initial_val_loss = initial_results[0].get('val/loss', 0.0)
            wandb_logger.experiment.log({
                'val/loss': initial_val_loss,
                'epoch': 0,
                'global_step': 0
            })
            print(f"Initial validation loss: {initial_val_loss:.6f}")
            
        print("===== Initial validation complete =====\n")
        
        # Now start training
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=args.resume_from if args.resume_from else None
        )
        
        # Save complete model checkpoint
        if args.checkpoint_dir:
            # Save complete checkpoint with adapter and FiLM parameters
            checkpoint_path = os.path.join(run_dir, "vocaldm_checkpoint_final.pt")
            save_vocaldm_checkpoint(
                model, 
                checkpoint_path,
                epoch=trainer.current_epoch,
                global_step=trainer.global_step
            )
            print(f"Saved complete checkpoint to {checkpoint_path}")
            
            # For compatibility, also save adapter-only state
            adapter_path = os.path.join(run_dir, "qvim_adapter.pt")
            torch.save(model.adapter.state_dict(), adapter_path)
            print(f"Saved adapter-only model to {adapter_path}")
            
            # Also save to original location for backward compatibility
            compat_path = os.path.join(args.checkpoint_dir, "qvim_adapter.pt")
            torch.save(model.adapter.state_dict(), compat_path)
            
            # Log the models to wandb - only on rank 0
            if is_global_zero() and wandb_logger and wandb_logger.experiment:
                try:
                    # Upload complete checkpoint
                    wandb_logger.experiment.log_artifact(
                        checkpoint_path, 
                        name="vocaldm_checkpoint_final", 
                        type="model"
                    )
                    print(f"Uploaded complete checkpoint to wandb")
                    
                    # Also upload adapter-only for compatibility
                    wandb_logger.experiment.log_artifact(
                        adapter_path, 
                        name="qvim_adapter", 
                        type="model"
                    )
                except Exception as e:
                    print(f"Failed to upload checkpoint to wandb: {e}")
                
        print(f"\nTraining completed successfully!")
        
        # Prompt user to save full model
        save_full = input("\nDo you want to save a copy of the full model? This will be large (~4GB) [y/N]: ").strip().lower()
        if save_full == 'y' or save_full == 'yes':
            if args.checkpoint_dir:
                # Create full model directory
                full_model_path = os.path.join(run_dir, "full_model.pt")
                print(f"\nSaving full model to {full_model_path}...")
                
                # Save entire model
                full_state = {
                    'audioldm': model.audioldm.state_dict(),
                    'adapter': model.adapter.state_dict(),
                    'qvim_model': model.qvim_model.state_dict(),
                    'epoch': trainer.current_epoch,
                    'global_step': trainer.global_step,
                    'hparams': model.hparams
                }
                torch.save(full_state, full_model_path)
                print(f"Full model saved successfully!")
            else:
                print("Cannot save full model without a checkpoint directory.")
        else:
            print("Full model save skipped.")
        return True
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")
        # Save current model state on interrupt
        if args.checkpoint_dir and hasattr(model, 'adapter'):
            # Save complete interrupted checkpoint
            interrupt_path = os.path.join(run_dir, "vocaldm_checkpoint_interrupted.pt")
            save_vocaldm_checkpoint(
                model, 
                interrupt_path,
                epoch=trainer.current_epoch if hasattr(trainer, 'current_epoch') else None,
                global_step=trainer.global_step if hasattr(trainer, 'global_step') else None
            )
            print(f"Saved interrupted checkpoint to {interrupt_path}")
            
            # For compatibility, also save adapter-only 
            adapter_path = os.path.join(run_dir, "qvim_adapter_interrupted.pt")
            torch.save(model.adapter.state_dict(), adapter_path)
            
            # Also save to original location
            compat_path = os.path.join(args.checkpoint_dir, "qvim_adapter_interrupted.pt")
            torch.save(model.adapter.state_dict(), compat_path)
            print(f"Also saved adapter-only state to {compat_path}")
        return False
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up resources
        cleanup_resources()
        
        # Close wandb run properly - only on rank 0
        if is_global_zero() and wandb_logger and wandb_logger.experiment:
            wandb_logger.experiment.finish()
            print("Wandb logging finalized")

class GuidanceScaleScheduler(pl.Callback):
    """Callback to gradually increase guidance scale during training"""
    def __init__(self, initial_scale=1.0, target_scale=3.0, warmup_percent=0.1, rampup_percent=0.4):
        super().__init__()
        self.initial_scale = initial_scale
        self.target_scale = target_scale
        self.warmup_percent = warmup_percent
        self.rampup_percent = rampup_percent
    
    def on_epoch_start(self, trainer, pl_module):
        # Calculate current guidance scale based on training progress
        progress = trainer.current_epoch / trainer.max_epochs
        
        if progress < self.warmup_percent:
            # Initial phase: use low guidance scale
            guidance_scale = self.initial_scale
        elif progress < (self.warmup_percent + self.rampup_percent):
            # Ramp-up phase: linearly increase guidance scale
            ramp_progress = (progress - self.warmup_percent) / self.rampup_percent
            guidance_scale = self.initial_scale + ramp_progress * (self.target_scale - self.initial_scale)
        else:
            # Final phase: use target guidance scale
            guidance_scale = self.target_scale
            
        # Update the model's guidance scale
        pl_module.config.guidance_scale = guidance_scale
        trainer.logger.experiment.log({"guidance_scale": guidance_scale})
        print(f"Epoch {trainer.current_epoch}: Setting guidance scale to {guidance_scale:.2f}")

if __name__ == "__main__":
    # Set up resource cleanup on exit
    import atexit
    atexit.register(cleanup_resources)
    
    parser = argparse.ArgumentParser(description="Train AudioLDM with QVIM conditioning")
    
    # General
    parser.add_argument("--project", type=str, default="vocaldm", help="Project name for wandb")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for wandb")
    parser.add_argument("--seed", type=int, default=46, help="Random seed for reproducibility")
    parser.add_argument("--debug_autograd", action="store_true", help="Enable autograd anomaly detection")
    parser.add_argument("--save_autograd_graph", action="store_true", help="Save autograd graph for debugging")
    
    # Paths
    parser.add_argument("--audioldm_model", type=str, default="audioldm-m-full", 
                       choices=["audioldm-m-full", "audioldm-s-full", "audioldm-s-full-v2", "audioldm-s-text-ft", "audioldm-m-text-ft", "audioldm-l-full"],
                       help="AudioLDM model name to use from Hugging Face Hub")
    parser.add_argument("--disable_checkpointing", action="store_true", default=True,
                       help="Disable gradient checkpointing to avoid gradient issues")
    parser.add_argument("--audioldm_checkpoint", type=str, default=None, 
                       help="Path to AudioLDM checkpoint (if not using model name)")
    parser.add_argument("--qvim_checkpoint", type=str, default="audioldm/qvim/baseline-ckpt/baseline.ckpt", 
                       help="Path to QVIM checkpoint")
    parser.add_argument("--dataset_path", type=str, default="audioldm/qvim/data", 
                       help="Path to dataset directory")
    parser.add_argument("--checkpoint_dir", type=str, default="ckpt/vocaldm", 
                       help="Directory to save checkpoints")
    parser.add_argument("--resume_from", type=str, default=None, 
                       help="Path to checkpoint to resume from")
    
    # Model configuration
    parser.add_argument("--qvim_dim", type=int, default=960, help="QVIM embedding dimension")
    parser.add_argument("--audioldm_dim", type=int, default=512, help="AudioLDM conditioing embedding dimension")
    
    # Training parameters
    parser.add_argument("--max_epochs", type=int, default=300, help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=96, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--adapter_lr", type=float, default=1e-5, help="Learning rate for adapter")
    parser.add_argument("--film_lr", type=float, default=5e-6, help="Learning rate for FiLM conditioning layers")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--adapter_weight_decay", type=float, default=1e-3, help="Weight decay for adapter (higher for newly initialized components)")
    parser.add_argument("--film_weight_decay", type=float, default=1e-5, help="Weight decay for FiLM layers (lower for pre-trained components)")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "plateau", "none"], help="LR scheduler type")
    parser.add_argument("--lr_rampdown_epochs", type=int, default=30, help="Number of epochs over which to ramp down the learning rate (if using cosine)")
    parser.add_argument("--patience", type=int, default=8, help="Patience for early stopping")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--val_check_interval", type=float, default=1.0, help="Validation check interval (fraction of epoch or integer steps)")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--param_names_contain", type=str, nargs='*', default=['film', 'emb_layers'], help="Parameter names to include in training (space-separated). Use --param_names_contain '' to train only the adapter.")

    # Data processing
    parser.add_argument("--sample_rate", type=int, default=32000, help="Audio sample rate for QVIM (32kHz)")
    parser.add_argument("--audioldm_sample_rate", type=int, default=16000, help="Audio sample rate for AudioLDM (16kHz)")
    parser.add_argument("--duration", type=float, default=10.0, help="Audio duration in seconds (AudioLDM expects 10s)")
    parser.add_argument("--num_workers", type=int, default=24, help="Number of data loader workers")
    parser.add_argument("--pin_memory", action="store_true", default=True, help="Pin memory in DataLoader (faster but uses more RAM)")
    
    # Generation parameters
    parser.add_argument("--ddim_steps", type=int, default=100, help="Number of DDIM sampling steps")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="Classifier-free guidance scale")
    parser.add_argument("--initial_guidance_scale", type=float, default=1.0, help="Initial classifier-free guidance scale (for scheduling)")
    parser.add_argument("--use_guidance_scale_scheduler", action="store_true", default=True, help="Enable progressive guidance scale scheduling")
    parser.add_argument("--guidance_warmup_percent", type=float, default=0.1, help="Percentage of training to use initial guidance scale")
    parser.add_argument("--guidance_rampup_percent", type=float, default=0.4, help="Percentage of training for linear increase to target guidance scale")
    parser.add_argument("--generate_every_n_epochs", type=int, default=1, help="Generate samples every N epochs")
    parser.add_argument("--log_audio", action="store_true", default=True, help="Enable audio sample logging to WandB")
    
    # Debug/Development
    parser.add_argument("--max_items", type=int, default=None, help="Maximum number of items to use (for debugging)")
    parser.add_argument("--wandb_log_model", action="store_true", help="Automatically upload model checkpoints to WandB (can be large files)")
    parser.add_argument("--num_gpus", type=str, default="auto", help="Number of GPUs to use for training ('auto' or specific number)")
    
    args = parser.parse_args()
    
    # Validate input arguments
    # Check file paths
    if not os.path.exists(args.qvim_checkpoint):
        raise FileNotFoundError(f"QVIM checkpoint not found: {args.qvim_checkpoint}")
    
    # Only check audioldm_checkpoint if it's not None (user could be using model name instead)
    if args.audioldm_checkpoint is not None and not os.path.exists(args.audioldm_checkpoint):
        raise FileNotFoundError(f"AudioLDM checkpoint not found: {args.audioldm_checkpoint}")
    
    # Validate numeric parameters
    if args.batch_size <= 0:
        raise ValueError(f"Batch size must be positive, got {args.batch_size}")
    
    if args.adapter_lr <= 0 or args.film_lr <= 0:
        raise ValueError(f"Learning rates must be positive, got adapter_lr={args.adapter_lr}, film_lr={args.film_lr}")
    
    if args.val_split <= 0 or args.val_split >= 1:
        raise ValueError(f"Validation split must be between 0 and 1, got {args.val_split}")
        
    # Ensure QVIM-specific parameters are correct
    if args.sample_rate != 32000:
        print(f"Warning: QVIM expects 32kHz sample rate, but {args.sample_rate}Hz was specified.")
        print("This may impact QVIM embedding quality.")
        
    # Ensure AudioLDM-specific parameters are correct
    if args.audioldm_sample_rate != 16000:
        print(f"Warning: AudioLDM expects 16kHz sample rate, but {args.audioldm_sample_rate}Hz was specified.")
        print("This may impact AudioLDM performance.")
        
    if args.duration != 10.0:
        print(f"Warning: AudioLDM was trained on 10-second audio, but {args.duration}s was specified.")
        print("Continuing with specified duration, but this may impact performance.")
    
    # Create checkpoint directory if it doesn't exist
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"Checkpoints will be saved to {args.checkpoint_dir}")
    
    print(f"Starting training with configuration:")
    print(f"  QVIM checkpoint: {args.qvim_checkpoint}")
    print(f"  AudioLDM checkpoint: {args.audioldm_checkpoint}")
    print(f"  Dataset path: {args.dataset_path}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rates - Adapter: {args.adapter_lr}, Film: {args.film_lr}")
    
    # Start training and get result
    success = train_vocaldm(args)
    
    # Exit with appropriate code
    if not success:
        print("Training did not complete successfully.")
        import sys
        sys.exit(1)
    else:
        print("Training completed successfully!")
        sys.exit(0)