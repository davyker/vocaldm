#!/usr/bin/env python
import argparse
import os
import math
import copy
import warnings

import torch
import torch.nn.functional as F  # Added for explicit operations
import torchaudio  # Added for explicit imports
import numpy as np
import pytorch_lightning as pl

# Enable Tensor Cores for faster training with minimal precision loss
torch.set_float32_matmul_precision('high')

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor

# Import from QVIM
from audioldm.qvim.src.qvim_mn_baseline.dataset import VimSketchDataset, AESAIMLA_DEV
from audioldm.qvim.src.qvim_mn_baseline.download import download_vimsketch_dataset, download_qvim_dev_dataset
from audioldm.qvim.src.qvim_mn_baseline.mn.model import get_model as get_mobilenet
from audioldm.qvim.src.qvim_mn_baseline.utils import NAME_TO_WIDTH
from audioldm.qvim.src.qvim_mn_baseline.ex_qvim import QVIMModule, train as qvim_train

# Import CLAP components
from audioldm.clap.training.data import get_audio_features

# Silence warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

class QVIMCLAPModule(QVIMModule):
    """
    Pytorch Lightning Module for QVIM-CLAP alignment
    Extends the original QVIMModule with direct CLAP alignment objectives
    """

    def __init__(self, config, clap_model=None):
        # Initialize parent QVIMModule with 512d output directly
        # Override pretrained_name to use our own modified copy that outputs 512d
        config.output_dim = 512  # Add this attribute to config
        super().__init__(config)
        
        # Store config for debug flags
        self.config = config
        
        # Custom initialization of QVIM encoders with direct 512d output for CLAP alignment
        # Using custom weight loading to handle dimension mismatch
        self.imitation_encoder = self._create_mobilenet_with_custom_loading(
            config.pretrained_name,
            NAME_TO_WIDTH(config.pretrained_name),
            output_dim=512
        )
        self.reference_encoder = copy.deepcopy(self.imitation_encoder)
        
        # Initialize cross-model temperature parameter (using same approach as tau)
        initial_cross_temp = torch.zeros((1,)) + config.initial_tau
        self.cross_temp = torch.nn.Parameter(initial_cross_temp, requires_grad=True)
        
        # Use the provided CLAP model from AudioLDM
        if clap_model is None:
            raise ValueError("CLAP model must be provided. It should be extracted from AudioLDM.")
            
        self.clap_model = clap_model
        
        # Make sure model is in eval mode
        self.clap_model.eval()
        
        # Ensure CLAP model is frozen (no gradient updates)
        for param in self.clap_model.parameters():
            param.requires_grad = False
            
        # Cache for validation batch CLAP embeddings
        self.val_clap_cache = {}
            
        # Initialize embedding caches for training and validation
        # Use dictionaries with filename hashes as keys
        self.clap_train_cache = {}
        self.clap_val_cache = {}
            
        # Register hooks to monitor hidden layer activations
        if hasattr(self.config, 'debug') and self.config.debug:
            self.activation_hooks = []
            
            def hook_fn(name):
                def hook(module, input, output):
                    # Handle tuple outputs (from some layers that return multiple values)
                    if isinstance(output, tuple):
                        x = output[0]  # Get the first element (usually the main tensor)
                    else:
                        x = output
                        
                    # Print statistics
                    print(f"[HIDDEN LAYER] {name}: shape={x.shape}, "
                          f"min={x.min().item():.4f}, max={x.max().item():.4f}, "
                          f"mean={x.mean().item():.4f}, std={x.std().item():.4f}")
                    
                    # Return original output unchanged
                    return output
                return hook
            
            # Add hooks to audio branch layers
            if hasattr(self.clap_model.model, 'audio_branch'):
                audio_branch = self.clap_model.model.audio_branch
                
                # Monitor all significant components in the audio processing pipeline
                
                # Spectrogram extraction
                if hasattr(audio_branch, 'spectrogram_extractor'):
                    self.activation_hooks.append(
                        audio_branch.spectrogram_extractor.register_forward_hook(hook_fn('spectrogram_extractor'))
                    )
                
                # Log-mel extraction
                if hasattr(audio_branch, 'logmel_extractor'):
                    self.activation_hooks.append(
                        audio_branch.logmel_extractor.register_forward_hook(hook_fn('logmel_extractor'))
                    )
                
                # Batch normalization
                if hasattr(audio_branch, 'bn0'):
                    self.activation_hooks.append(
                        audio_branch.bn0.register_forward_hook(hook_fn('bn0'))
                    )
                
                # Spectrogram augmentation
                if hasattr(audio_branch, 'spec_augmenter'):
                    self.activation_hooks.append(
                        audio_branch.spec_augmenter.register_forward_hook(hook_fn('spec_augmenter'))
                    )
                
                # Patch embedding
                if hasattr(audio_branch, 'patch_embed'):
                    self.activation_hooks.append(
                        audio_branch.patch_embed.register_forward_hook(hook_fn('patch_embed'))
                    )
                    
                    # Also hook into the projection layer
                    if hasattr(audio_branch.patch_embed, 'proj'):
                        self.activation_hooks.append(
                            audio_branch.patch_embed.proj.register_forward_hook(hook_fn('patch_embed.proj'))
                        )
                
                # Position embedding
                if hasattr(audio_branch, 'pos_drop'):
                    self.activation_hooks.append(
                        audio_branch.pos_drop.register_forward_hook(hook_fn('pos_drop'))
                    )
                
                # Transformer layers
                if hasattr(audio_branch, 'layers'):
                    for i, layer in enumerate(audio_branch.layers):
                        self.activation_hooks.append(
                            layer.register_forward_hook(hook_fn(f'layer_{i}'))
                        )
                        
                        # Add hooks for the transformer blocks within each layer
                        if hasattr(layer, 'blocks'):
                            for j, block in enumerate(layer.blocks):
                                self.activation_hooks.append(
                                    block.register_forward_hook(hook_fn(f'layer_{i}.block_{j}'))
                                )
                
                # Final normalization
                if hasattr(audio_branch, 'norm'):
                    self.activation_hooks.append(
                        audio_branch.norm.register_forward_hook(hook_fn('final_norm'))
                    )
                    
                # Average pooling
                if hasattr(audio_branch, 'avgpool'):
                    self.activation_hooks.append(
                        audio_branch.avgpool.register_forward_hook(hook_fn('avgpool'))
                    )
                
                print(f"[CLAP SETUP] Registered {len(self.activation_hooks)} hooks for hidden layer monitoring")
            
        # Debug info
        if hasattr(self.config, 'debug') and self.config.debug:
            print(f"[CLAP SETUP] Using AudioLDM CLAP model: {self.clap_model.__class__.__name__}")
            if hasattr(self.clap_model, 'model') and hasattr(self.clap_model.model, 'audio_branch'):
                print(f"[CLAP SETUP] Audio branch type: {self.clap_model.model.audio_branch.__class__.__name__}")
                print(f"[CLAP SETUP] Audio config: {self.clap_model.model_cfg['audio_cfg'] if hasattr(self.clap_model, 'model_cfg') else 'N/A'}")
            print(f"[CLAP SETUP] CLAP embed mode: {self.clap_model.embed_mode}")
    
    def _create_mobilenet_with_custom_loading(self, pretrained_name, width_mult, output_dim=512):
        """
        Create a MobileNetV3 model with custom dimension output and handle weight loading
        to avoid dimension mismatch errors with pre-trained weights.
        """
        from audioldm.qvim.src.qvim_mn_baseline.mn.model import (
            mobilenet_v3, _mobilenet_v3_conf, MobileNetV3, 
            pretrained_models, model_dir
        )
        
        # Get configuration but don't load pretrained weights yet
        inverted_residual_setting, last_channel = _mobilenet_v3_conf(
            width_mult=width_mult,
            reduced_tail=False,
            dilated=False,
            strides=(2, 2, 2, 2)
        )
        
        # Model arguments with custom output dimension
        model_args = {
            'head_type': 'mlp',
            'num_classes': 527,
            'multihead_attention_heads': 4,
            'input_dims': (128, 1000),
            'se_conf': {'se_dims': None},
            'output_dim': output_dim
        }
        
        # Create model with our custom dimension
        model = MobileNetV3(inverted_residual_setting, last_channel, **model_args)
        
        # Custom weight loading to handle dimension mismatch
        if pretrained_name in pretrained_models:
            from torch.hub import load_state_dict_from_url
            model_url = pretrained_models.get(pretrained_name)
            state_dict = load_state_dict_from_url(model_url, model_dir=model_dir, map_location="cpu")
            
            # Remove layers with dimension mismatch
            for key in ['features.16.0.weight', 'features.16.1.weight', 'features.16.1.bias', 
                       'features.16.1.running_mean', 'features.16.1.running_var', 'classifier.2.weight']:
                if key in state_dict:
                    del state_dict[key]
            
            # Load compatible weights
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pre-trained weights from {pretrained_name} with {output_dim}-dim output compatibility.")
        
        return model
        
    def forward_clap(self, audio):
        """Forward audio through CLAP to get embeddings"""
        # CLAP expects audio in range [-1, 1] at 16kHz
        with torch.no_grad():
            # ----------------- INPUT VALIDATION -----------------
            if hasattr(self.config, 'debug') and self.config.debug:
                print(f"[CLAP PIPELINE] Input audio shape: {audio.shape}")
                print(f"[CLAP PIPELINE] Input audio stats: min={audio.min().item():.4f}, max={audio.max().item():.4f}, mean={audio.mean().item():.4f}, std={audio.std().item():.4f}")
                nan_count = torch.isnan(audio).sum().item()
                inf_count = torch.isinf(audio).sum().item()
                if nan_count > 0 or inf_count > 0:
                    print(f"[CLAP PIPELINE] WARNING: Input contains {nan_count}/{audio.numel()} NaN values and {inf_count}/{audio.numel()} Inf values")
            
            # First get 2D tensor [batch_size, samples]
            if audio.dim() > 2:
                audio = audio.squeeze(1)
            
            # ----------------- RESAMPLING -----------------
            # Resample from 32kHz to 16kHz
            audio_16k = torchaudio.functional.resample(
                audio, 
                orig_freq=32000,  # QVIM sample rate 
                new_freq=16000    # CLAP required sample rate
            )
            
            if hasattr(self.config, 'debug') and self.config.debug:
                print(f"[CLAP PIPELINE] After resampling: shape={audio_16k.shape}")
                print(f"[CLAP PIPELINE] After resampling stats: min={audio_16k.min().item():.4f}, max={audio_16k.max().item():.4f}, mean={audio_16k.mean().item():.4f}, std={audio_16k.std().item():.4f}")
                
                nan_count = torch.isnan(audio_16k).sum().item()
                inf_count = torch.isinf(audio_16k).sum().item()
                if nan_count > 0 or inf_count > 0:
                    print(f"[CLAP PIPELINE] WARNING: After resampling: {nan_count}/{audio_16k.numel()} NaN values and {inf_count}/{audio_16k.numel()} Inf values")
            
            # ----------------- WAVEFORM PROCESSING -----------------
            # Process audio one sample at a time as CLAP expects
            
            # Save the current batch size
            batch_size = audio_16k.size(0)
            
            if hasattr(self.config, 'debug') and self.config.debug:
                print(f"[CLAP PIPELINE] Processing {batch_size} samples individually")
            
            # Process each audio sample individually and collect embeddings
            embeddings_list = []
            
            # Save original embed_mode
            original_embed_mode = self.clap_model.embed_mode
            
            # Force audio embedding mode
            self.clap_model.embed_mode = "audio"
            
            for i in range(batch_size):
                # Extract single waveform
                single_waveform = audio_16k[i:i+1]  # Keep batch dimension [1, samples]
                
                # Trim trailing zeros from the waveform
                epsilon = 1e-3  # Threshold for considering a value as "zero"
                waveform = single_waveform.squeeze(0)  # Remove batch dim for processing
                
                # Find where the trailing near-zeros start
                idx = len(waveform) - 1
                while idx >= 0 and torch.abs(waveform[idx]) < epsilon:
                    idx -= 1
                
                # Add 1 to include the last non-zero sample
                trimmed_length = idx + 1
                
                # Create trimmed waveform
                if trimmed_length < len(waveform):
                    trimmed_waveform = waveform[:trimmed_length]
                    # Add small Gaussian noise for numerical stability
                    trimmed_waveform = trimmed_waveform + torch.randn_like(trimmed_waveform) * 1e-5
                    single_waveform = trimmed_waveform.unsqueeze(0)  # Add batch dimension back
                    
                    if hasattr(self.config, 'debug') and self.config.debug:
                        print(f"[CLAP PIPELINE] Removed {len(waveform)-trimmed_length} trailing zeros from sample {i}")
                else:
                    # Add small Gaussian noise for numerical stability
                    waveform = waveform + torch.randn_like(waveform) * 1e-5
                    single_waveform = waveform.unsqueeze(0)  # Add batch dimension back
                    
                    if hasattr(self.config, 'debug') and self.config.debug:
                        print(f"[CLAP PIPELINE] Removed 0 trailing zeros from sample {i}")
                
                if hasattr(self.config, 'debug') and self.config.debug and i == 0:
                    print(f"[CLAP PIPELINE] Sample 0 shape: {single_waveform.shape}")
                    near_zero_count = (torch.abs(single_waveform.squeeze(0)) < epsilon).sum().item()
                    print(f"[CLAP PIPELINE] Sample 0: {near_zero_count}/{single_waveform.numel()} near-zero values")
                
                # Use AudioLDM's CLAP model to get audio embedding
                # First, we need to convert to the expected format (48kHz)
                audio_dict_list = []
                
                # This step is from AudioLDM's CLAP implementation - it resamples to 48kHz
                single_waveform_48k = torchaudio.functional.resample(
                    single_waveform, 
                    orig_freq=16000, 
                    new_freq=48000
                )
                
                # Convert to list format expected by get_audio_features
                waveforms = [w for w in self.clap_model.batch_to_list(single_waveform_48k)]
                
                # Create audio dictionary for each waveform
                for wf in waveforms:
                    audio_dict = {}
                    audio_dict = get_audio_features(
                        audio_dict,
                        wf,
                        480000,  # Max audio length
                        data_truncating="fusion",
                        data_filling="repeatpad",
                        audio_cfg=self.clap_model.model_cfg["audio_cfg"]
                    )
                    audio_dict_list.append(audio_dict)
                
                # Get audio embedding directly using the model's audio embedding method
                embedding = self.clap_model.model.get_audio_embedding(audio_dict_list)
                
                # Add to our list
                embeddings_list.append(embedding)
            
            # Restore embed_mode
            self.clap_model.embed_mode = original_embed_mode
            
            # Concatenate all embeddings into a single batch tensor
            clap_embedding = torch.cat(embeddings_list, dim=0)
            
            # Add an extra dimension to match expected shape [batch_size, 1, embed_dim]
            clap_embedding = clap_embedding.unsqueeze(1)
            
            if hasattr(self.config, 'debug') and self.config.debug:
                print(f"[CLAP PIPELINE] Final embeddings batch shape: {clap_embedding.shape}")
            
            # ----------------- EMBEDDING VALIDATION -----------------
            if hasattr(self.config, 'debug') and self.config.debug:
                print(f"[CLAP PIPELINE] CLAP embedding shape: {clap_embedding.shape}")
                
                nan_count = torch.isnan(clap_embedding).sum().item()
                inf_count = torch.isinf(clap_embedding).sum().item()
                
                if nan_count > 0 or inf_count > 0:
                    print(f"[CLAP PIPELINE] WARNING: CLAP embedding has {nan_count}/{clap_embedding.numel()} NaN values and {inf_count}/{clap_embedding.numel()} Inf values")
                else:
                    print(f"[CLAP PIPELINE] CLAP embedding seems valid (no NaN/Inf)")
                    
                print(f"[CLAP PIPELINE] CLAP embedding stats: min={clap_embedding.min().item():.4f}, max={clap_embedding.max().item():.4f}, mean={clap_embedding.mean().item():.4f}, std={clap_embedding.std().item():.4f}")
                
                # Check if any embeddings are all zeros
                zero_rows = (torch.abs(clap_embedding).sum(dim=(1,2)) < 1e-6).sum().item()
                if zero_rows > 0:
                    print(f"[CLAP PIPELINE] WARNING: {zero_rows}/{clap_embedding.shape[0]} CLAP embeddings are all zeros")
                
                # Check norms of embeddings
                norms = torch.norm(clap_embedding, dim=2).mean(dim=1)
                print(f"[CLAP PIPELINE] CLAP embedding norms: min={norms.min().item():.4f}, max={norms.max().item():.4f}, mean={norms.mean().item():.4f}")
                
        # Remove the middle dimension for compatibility with the rest of the code
        # [batch_size, 1, embed_dim] -> [batch_size, embed_dim]
        return clap_embedding.squeeze(1)

    def training_step(self, batch, batch_idx):
        # Store current batch for use in lr_scheduler_step
        self.current_batch = batch
        
        self.lr_scheduler_step(batch_idx)

        # Get QVIM embeddings for imitation and reference
        y_imitation, y_reference = self.forward(batch['imitation'], batch['reference'])
        
        # Get CLAP embeddings for reference
        y_clap = self.forward_clap(batch['reference'])
        
        # Debug embeddings right before computing similarities
        if hasattr(self.config, 'debug') and self.config.debug:
            print(f"[SIMILARITY] Shapes: y_im {y_imitation.shape}, y_ref {y_reference.shape}, y_clap {y_clap.shape}")
            print(f"[SIMILARITY] Norms: y_im {torch.norm(y_imitation, dim=1).mean().item():.4f}, "
                  f"y_ref {torch.norm(y_reference, dim=1).mean().item():.4f}, "
                  f"y_clap {torch.norm(y_clap, dim=1).mean().item():.4f}")
            print(f"[SIMILARITY] Temps: tau {self.tau.item():.6f}, cross_temp {self.cross_temp.item():.6f}")
            
            # Check for NaN/Inf in embeddings
            nan_im = torch.isnan(y_imitation).sum().item()
            nan_ref = torch.isnan(y_reference).sum().item()
            nan_clap = torch.isnan(y_clap).sum().item()
            inf_im = torch.isinf(y_imitation).sum().item()
            inf_ref = torch.isinf(y_reference).sum().item()
            inf_clap = torch.isinf(y_clap).sum().item()
            
            if nan_im > 0 or inf_im > 0:
                print(f"[SIMILARITY] WARNING: y_im has {nan_im}/{y_imitation.numel()} NaN and {inf_im}/{y_imitation.numel()} Inf")
            if nan_ref > 0 or inf_ref > 0:
                print(f"[SIMILARITY] WARNING: y_ref has {nan_ref}/{y_reference.numel()} NaN and {inf_ref}/{y_reference.numel()} Inf")
            if nan_clap > 0 or inf_clap > 0:
                print(f"[SIMILARITY] WARNING: y_clap has {nan_clap}/{y_clap.numel()} NaN and {inf_clap}/{y_clap.numel()} Inf")
        
        # Calculate batch similarity matrix for all three objectives
        
        # 1. QVIM internal similarity (imitation to reference)
        C_qvim = torch.matmul(y_imitation, y_reference.T)
        C_qvim = C_qvim / torch.abs(self.tau)
        C_qvim_log = torch.log_softmax(C_qvim, dim=1)
        
        # 2. QVIM reference to CLAP similarity
        C_ref_clap = torch.matmul(y_reference, y_clap.T)
        C_ref_clap = C_ref_clap / torch.abs(self.cross_temp)
        C_ref_clap_log = torch.log_softmax(C_ref_clap, dim=1)
        
        # 3. QVIM imitation to CLAP similarity
        C_im_clap = torch.matmul(y_imitation, y_clap.T)
        C_im_clap = C_im_clap / torch.abs(self.cross_temp)
        C_im_clap_log = torch.log_softmax(C_im_clap, dim=1)
        
        # Debug similarity matrices
        if hasattr(self.config, 'debug') and self.config.debug:
            print(f"[SIMILARITY] Similarity matrices: C_qvim {C_qvim.shape}, C_ref_clap {C_ref_clap.shape}, C_im_clap {C_im_clap.shape}")
            print(f"[SIMILARITY] C_qvim: min={C_qvim.min().item():.4f}, max={C_qvim.max().item():.4f}, mean={C_qvim.mean().item():.4f}")
            print(f"[SIMILARITY] C_ref_clap: min={C_ref_clap.min().item():.4f}, max={C_ref_clap.max().item():.4f}, mean={C_ref_clap.mean().item():.4f}")
            print(f"[SIMILARITY] C_im_clap: min={C_im_clap.min().item():.4f}, max={C_im_clap.max().item():.4f}, mean={C_im_clap.mean().item():.4f}")
            
            # Check log softmax outputs for NaN
            nan_qvim_log = torch.isnan(C_qvim_log).sum().item()
            nan_ref_clap_log = torch.isnan(C_ref_clap_log).sum().item()
            nan_im_clap_log = torch.isnan(C_im_clap_log).sum().item()
            
            if nan_qvim_log > 0:
                print(f"[SIMILARITY] WARNING: C_qvim_log has {nan_qvim_log}/{C_qvim_log.numel()} NaN values")
            if nan_ref_clap_log > 0:
                print(f"[SIMILARITY] WARNING: C_ref_clap_log has {nan_ref_clap_log}/{C_ref_clap_log.numel()} NaN values")
            if nan_im_clap_log > 0:
                print(f"[SIMILARITY] WARNING: C_im_clap_log has {nan_im_clap_log}/{C_im_clap_log.numel()} NaN values")
        
        # Create identity matrix based on audio filenames (same as original QVIM)
        paths = np.array([hash(p) for i, p in enumerate(batch['imitation_filename'])])
        I = torch.tensor(paths[None, :] == paths[:, None])

        # Calculate loss for each objective
        loss_qvim = -C_qvim_log[torch.where(I)].mean()
        loss_ref_clap = -C_ref_clap_log[torch.where(I)].mean()
        loss_im_clap = -C_im_clap_log[torch.where(I)].mean()
        
        # Combined loss (equal weighting)
        total_loss = (loss_qvim + loss_ref_clap + loss_im_clap) / 3.0

        # Log all losses
        self.log('train/loss', total_loss, prog_bar=True, batch_size=len(batch['imitation']))
        self.log('train/loss_qvim', loss_qvim, batch_size=len(batch['imitation']))
        self.log('train/loss_ref_clap', loss_ref_clap, batch_size=len(batch['imitation']))
        self.log('train/loss_im_clap', loss_im_clap, batch_size=len(batch['imitation']))
        self.log('train/tau', self.tau, batch_size=len(batch['imitation']))
        self.log('train/cross_temp', self.cross_temp, batch_size=len(batch['imitation']))
        self.log('lr', self.optimizers().param_groups[0]['lr'], prog_bar=True, batch_size=len(batch['imitation']))

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Debug validation batch
        if hasattr(self.config, 'debug') and self.config.debug:
            print(f"\n[VALIDATION] Batch {batch_idx} - Processing validation data")
            
            # Check input audio
            if 'imitation' in batch and 'reference' in batch:
                imitation = batch['imitation']
                reference = batch['reference']
                print(f"[VALIDATION] Imitation shape: {imitation.shape}, Reference shape: {reference.shape}")
                print(f"[VALIDATION] Imitation: min={imitation.min().item():.4f}, max={imitation.max().item():.4f}")
                print(f"[VALIDATION] Reference: min={reference.min().item():.4f}, max={reference.max().item():.4f}")
                
                # Check for NaN/Inf in input
                nan_im = torch.isnan(imitation).sum().item()
                nan_ref = torch.isnan(reference).sum().item()
                inf_im = torch.isinf(imitation).sum().item()
                inf_ref = torch.isinf(reference).sum().item()
                
                if nan_im > 0 or inf_im > 0:
                    print(f"[VALIDATION] WARNING: Imitation has {nan_im}/{imitation.numel()} NaN and {inf_im}/{imitation.numel()} Inf")
                if nan_ref > 0 or inf_ref > 0:
                    print(f"[VALIDATION] WARNING: Reference has {nan_ref}/{reference.numel()} NaN and {inf_ref}/{reference.numel()} Inf")
        
        # Get QVIM embeddings
        y_imitation, y_reference = self.forward(batch['imitation'], batch['reference'])
        
        # Get CLAP embeddings (using cache if available)
        if batch_idx in self.val_clap_cache:
            # Use cached embeddings
            y_clap = self.val_clap_cache[batch_idx].to(y_reference.device)
            if hasattr(self.config, 'debug') and self.config.debug:
                print(f"[VALIDATION] Using cached CLAP embeddings for batch {batch_idx}")
        else:
            # Generate new embeddings and cache them
            y_clap = self.forward_clap(batch['reference'])
            self.val_clap_cache[batch_idx] = y_clap.detach().cpu()  # Store on CPU to save GPU memory
            if hasattr(self.config, 'debug') and self.config.debug:
                print(f"[VALIDATION] Generated and cached new CLAP embeddings for batch {batch_idx}")
        
        # Debug embeddings
        if hasattr(self.config, 'debug') and self.config.debug:
            print(f"[VALIDATION] Shapes: y_im {y_imitation.shape}, y_ref {y_reference.shape}, y_clap {y_clap.shape}")
            print(f"[VALIDATION] Norms: y_im {torch.norm(y_imitation, dim=1).mean().item():.4f}, "
                  f"y_ref {torch.norm(y_reference, dim=1).mean().item():.4f}, "
                  f"y_clap {torch.norm(y_clap, dim=1).mean().item():.4f}")
                  
            # Check for NaN/Inf in embeddings
            nan_im = torch.isnan(y_imitation).sum().item()
            nan_ref = torch.isnan(y_reference).sum().item()
            nan_clap = torch.isnan(y_clap).sum().item()
            inf_im = torch.isinf(y_imitation).sum().item()
            inf_ref = torch.isinf(y_reference).sum().item()
            inf_clap = torch.isinf(y_clap).sum().item()
            
            if nan_im > 0 or inf_im > 0:
                print(f"[VALIDATION] WARNING: y_im has {nan_im}/{y_imitation.numel()} NaN and {inf_im}/{y_imitation.numel()} Inf")
            if nan_ref > 0 or inf_ref > 0:
                print(f"[VALIDATION] WARNING: y_ref has {nan_ref}/{y_reference.numel()} NaN and {inf_ref}/{y_reference.numel()} Inf")
            if nan_clap > 0 or inf_clap > 0:
                print(f"[VALIDATION] WARNING: y_clap has {nan_clap}/{y_clap.numel()} NaN and {inf_clap}/{y_clap.numel()} Inf")
        
        # Calculate batch similarity matrix for all three objectives
        
        # 1. QVIM internal similarity (imitation to reference)
        C_qvim = torch.matmul(y_imitation, y_reference.T)
        C_qvim = C_qvim / torch.abs(self.tau)
        C_qvim_log = torch.log_softmax(C_qvim, dim=1)
        
        # 2. QVIM reference to CLAP similarity
        C_ref_clap = torch.matmul(y_reference, y_clap.T)
        C_ref_clap = C_ref_clap / torch.abs(self.cross_temp)
        C_ref_clap_log = torch.log_softmax(C_ref_clap, dim=1)
        
        # 3. QVIM imitation to CLAP similarity
        C_im_clap = torch.matmul(y_imitation, y_clap.T)
        C_im_clap = C_im_clap / torch.abs(self.cross_temp)
        C_im_clap_log = torch.log_softmax(C_im_clap, dim=1)
        
        # Debug similarity matrices
        if hasattr(self.config, 'debug') and self.config.debug:
            # Check for NaN in log_softmax outputs
            nan_qvim_log = torch.isnan(C_qvim_log).sum().item()
            nan_ref_clap_log = torch.isnan(C_ref_clap_log).sum().item()
            nan_im_clap_log = torch.isnan(C_im_clap_log).sum().item()
            
            if nan_qvim_log > 0 or nan_ref_clap_log > 0 or nan_im_clap_log > 0:
                print(f"[VALIDATION] WARNING: NaN found in log_softmax outputs:")
                print(f"[VALIDATION]   - C_qvim_log: {nan_qvim_log}/{C_qvim_log.numel()} NaN values")
                print(f"[VALIDATION]   - C_ref_clap_log: {nan_ref_clap_log}/{C_ref_clap_log.numel()} NaN values")
                print(f"[VALIDATION]   - C_im_clap_log: {nan_im_clap_log}/{C_im_clap_log.numel()} NaN values")
        
        # Identity matrix based on audio filenames
        paths = np.array([hash(p) for i, p in enumerate(batch['imitation_filename'])])
        I = torch.tensor(paths[None, :] == paths[:, None])

        # Calculate loss for each objective
        loss_qvim = -C_qvim_log[torch.where(I)].mean()
        loss_ref_clap = -C_ref_clap_log[torch.where(I)].mean()
        loss_im_clap = -C_im_clap_log[torch.where(I)].mean()
        
        # Combined loss (equal weighting)
        total_loss = (loss_qvim + loss_ref_clap + loss_im_clap) / 3.0

        # Log all losses
        self.log('val/loss', total_loss, prog_bar=True, batch_size=len(batch['imitation']))
        self.log('val/loss_qvim', loss_qvim, batch_size=len(batch['imitation']))
        self.log('val/loss_ref_clap', loss_ref_clap, batch_size=len(batch['imitation']))
        self.log('val/loss_im_clap', loss_im_clap, batch_size=len(batch['imitation']))
        self.log('val/tau', self.tau, batch_size=len(batch['imitation']))
        self.log('val/cross_temp', self.cross_temp, batch_size=len(batch['imitation']))

        # Store data for MRR calculation (use QVIM internal similarity)
        self.validation_output.extend([
            {
                'imitation': copy.deepcopy(y_imitation.detach().cpu().numpy()),
                'reference': copy.deepcopy(y_reference.detach().cpu().numpy()),
                'imitation_filename': batch['imitation_filename'],
                'reference_filename': batch['reference_filename'],
                'imitation_class': batch['imitation_class'],
                'reference_class': batch['reference_class']
            }
        ])


def train(config):
    # Extend config with needed CLAP attributes 
    setattr(config, 'output_dim', 512)
    
    # Add pin_memory attribute required by ex_qvim.py
    if not hasattr(config, 'pin_memory'):
        setattr(config, 'pin_memory', True)
    
    print(f"Loading AudioLDM model: {config.audioldm_model}")
    
    # Import necessary modules
    from audioldm.pipeline import build_model
    
    # Build the AudioLDM model
    try:
        audioldm_model = build_model(model_name=config.audioldm_model)
        print(f"AudioLDM model loaded successfully")
    except Exception as e:
        print(f"Error loading AudioLDM model: {e}")
        exit(1)
    
    # Extract the CLAP model from AudioLDM
    clap_model = audioldm_model.cond_stage_model
    if clap_model is None:
        print("Failed to extract CLAP model from AudioLDM")
        exit(1)
        
    print(f"Extracted CLAP model: {clap_model.__class__.__name__}")
    if hasattr(config, 'debug') and config.debug:
        if hasattr(clap_model, 'model') and hasattr(clap_model.model, 'audio_branch'):
            print(f"CLAP audio branch: {clap_model.model.audio_branch.__class__.__name__}")
    
    # Use the existing train function with our custom module
    # Create a factory function to instantiate our module with the extracted CLAP model
    def model_factory(config):
        return QVIMCLAPModule(config, clap_model=clap_model)
        
    # Call the existing training function with our factory function
    qvim_train(config, model_factory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Argument parser for training the QVIM-CLAP alignment model.")

    # General
    parser.add_argument('--project', type=str, default="qvim-clap-alignment",
                        help="Project name in wandb.")
    parser.add_argument('--num_workers', type=int, default=16,
                        help="Number of data loader workers.")
    parser.add_argument('--pin_memory', type=bool, default=True,
                        help="Whether to pin memory for data loading (faster but uses more memory).")
    parser.add_argument('--model_save_path', type=str, default="audioldm/qvim/models_clap_aligned",
                        help="Path to store the checkpoints.")
    parser.add_argument('--dataset_path', type=str, default='audioldm/qvim/data',
                        help="Path to the data sets.")

    # AudioLDM Configuration
    parser.add_argument('--audioldm_model', type=str, default="audioldm-m-full",
                        choices=["audioldm-s-full", "audioldm-m-full", "audioldm-l-full", 
                                "audioldm-s-full-v2", "audioldm-m-text-ft", "audioldm-s-text-ft"],
                        help="AudioLDM model to use for alignment. CLAP will be taken directly from this model.")

    # Encoder architecture
    parser.add_argument('--pretrained_name', type=str, default="mn10_as",
                        help="Pretrained model name for transfer learning.")

    # Training
    parser.add_argument('--random_seed', type=int, default=42,
                        help="A seed to make the experiment reproducible.")
    parser.add_argument('--continue_from', type=str, default=None,
                        help="Path to checkpoint file to continue training from")
    parser.add_argument('--final_eval_dataset', type=str, default="val", choices=["dev", "val"],
                        help="Dataset to use for final evaluation: 'dev' (QVIM-DEV) or 'val' (VimSketch val split)")
    parser.add_argument('--val_split', type=float, default=0.15,
                        help="Fraction of the dataset to use for validation.")    
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Number of samples per batch.")
    parser.add_argument('--n_epochs', type=int, default=100,
                        help="Maximum number of training epochs (can stop earlier with early stopping).")
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0,
                        help="Minimum change in the monitored metric to qualify as an improvement.")
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help="L2 weight regularization to prevent overfitting.")
    parser.add_argument('--max_lr', type=float, default=0.0003,
                        help="Maximum learning rate.")
    parser.add_argument('--min_lr', type=float, default=0.000025,
                        help="Final learning rate at the end of training.")
    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help="Number of warm-up epochs where learning rate increases gradually.")
    parser.add_argument('--rampdown_epochs', type=int, default=22,
                        help="Duration (in epochs) for learning rate ramp-down.")
    parser.add_argument('--initial_tau', type=float, default=0.07,
                        help="Temperature parameter for the QVIM loss function and cross-temp loss.")
    parser.add_argument('--tau_trainable', default=True, action='store_true',
                        help="make tau trainable or not.")
    parser.add_argument('--lr_schedule', type=str, default="cosine", choices=["cosine", "plateau", "cosine_annealing"],
                        help="Learning rate schedule: 'cosine' (original), 'plateau' (reduce on plateau), or 'cosine_annealing' (smoother decay)")
    
    # Debug flags
    parser.add_argument('--debug', action='store_true',
                        help="Enable debug printing to diagnose NaN issues in both pipelines")
    parser.add_argument('--max_items', type=int, default=None,
                        help="Limit dataset to specified number of items for faster debugging")

    # Preprocessing
    parser.add_argument('--duration', type=float, default=10.0,
                        help="Duration of audio clips in seconds.")

    # Spectrogram Parameters
    parser.add_argument('--sample_rate', type=int, default=32000,
                        help="Target sampling rate for audio resampling.")
    parser.add_argument('--window_size', type=int, default=800,
                        help="Size of the window for STFT in samples.")
    parser.add_argument('--hop_size', type=int, default=320,
                        help="Hop length for STFT in samples.")
    parser.add_argument('--n_fft', type=int, default=1024,
                        help="Number of FFT bins for spectral analysis.")
    parser.add_argument('--n_mels', type=int, default=128,
                        help="Number of mel filter banks for Mel spectrogram conversion.")
    parser.add_argument('--freqm', type=int, default=8,
                        help="Frequency masking parameter for spectrogram augmentation.")
    parser.add_argument('--timem', type=int, default=300,
                        help="Time masking parameter for spectrogram augmentation.")
    parser.add_argument('--fmin', type=int, default=0,
                        help="Minimum frequency cutoff for Mel spectrogram.")
    parser.add_argument('--fmax', type=int, default=None,
                        help="Maximum frequency cutoff for Mel spectrogram (None means use Nyquist frequency).")
    parser.add_argument('--fmin_aug_range', type=int, default=10,
                        help="Variation range for fmin augmentation.")
    parser.add_argument('--fmax_aug_range', type=int, default=2000,
                        help="Variation range for fmax augmentation.")

    args = parser.parse_args()

    if args.random_seed:
        pl.seed_everything(args.random_seed)

    train(args)