#!/usr/bin/env python
import argparse
import os
import math
import copy
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import torchaudio

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
from audioldm.clap.encoders import CLAPAudioEmbeddingClassifierFreev2

# Silence warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

class QVIMCLAPModule(QVIMModule):
    """
    Pytorch Lightning Module for QVIM-CLAP alignment
    Extends the original QVIMModule with direct CLAP alignment objectives
    """

    def __init__(self, config):
        # Initialize parent QVIMModule with 512d output directly
        # Override pretrained_name to use our own modified copy that outputs 512d
        config.output_dim = 512  # Add this attribute to config
        super().__init__(config)
        
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
        
        # Load CLAP model (frozen) - MUST use 16kHz due to hard-coded assertion
        self.clap_model = CLAPAudioEmbeddingClassifierFreev2(
            pretrained_path=config.clap_checkpoint,
            sampling_rate=16000,  # CLAP requires exactly 16kHz - hard assertion in encoders.py
            embed_mode="audio",
            amodel=config.clap_model,
            unconditional_prob=0.0  # No need for unconditional samples during training
        )
        
        # Freeze CLAP model
        for param in self.clap_model.parameters():
            param.requires_grad = False
            
        self.clap_model.eval()
    
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
        
    def compute_safe_similarities(self, y_imitation, y_reference, y_clap):
        """Compute similarities between embeddings with safety checks to prevent NaN"""
        # First log/debug info to diagnose NaN issues
        print(f"DEBUG - Shapes: y_ref {y_reference.shape}, y_im {y_imitation.shape}, y_clap {y_clap.shape}")
        print(f"DEBUG - Norms: y_ref {torch.norm(y_reference, dim=1).mean()}, "
              f"y_im {torch.norm(y_imitation, dim=1).mean()}, "
              f"y_clap {torch.norm(y_clap, dim=1).mean()}")
        print(f"DEBUG - temps: tau {self.tau.item()}, cross_temp {self.cross_temp.item()}")
        
        # Re-normalize all embeddings for extra safety
        y_reference = F.normalize(y_reference, p=2, dim=1, eps=1e-8)
        y_imitation = F.normalize(y_imitation, p=2, dim=1, eps=1e-8)
        y_clap = F.normalize(y_clap, p=2, dim=1, eps=1e-8)
        
        # Safe temperature values (minimum 1e-4)
        safe_tau = torch.clamp(torch.abs(self.tau), min=1e-4)
        safe_cross_temp = torch.clamp(torch.abs(self.cross_temp), min=1e-4)
        
        # 1. QVIM internal similarity (imitation to reference)
        C_qvim = torch.matmul(y_imitation, y_reference.T)
        C_qvim = C_qvim / safe_tau
        C_qvim_log = F.log_softmax(C_qvim, dim=1)
        
        # 2. QVIM reference to CLAP similarity
        C_ref_clap = torch.matmul(y_reference, y_clap.T)
        C_ref_clap = C_ref_clap / safe_cross_temp
        C_ref_clap_log = F.log_softmax(C_ref_clap, dim=1)
        
        # 3. QVIM imitation to CLAP similarity
        C_im_clap = torch.matmul(y_imitation, y_clap.T)
        C_im_clap = C_im_clap / safe_cross_temp
        C_im_clap_log = F.log_softmax(C_im_clap, dim=1)
        
        # Check for NaN and provide fallback
        if torch.isnan(C_ref_clap_log).any() or torch.isnan(C_im_clap_log).any() or torch.isnan(C_qvim_log).any():
            print("WARNING: NaN detected in similarity computations")
            
            # Create backup identity matrix for safe loss computation
            batch_size = y_imitation.shape[0]
            I_fallback = torch.eye(batch_size, device=y_imitation.device)
            
            # Compute backup loss (simple cross-entropy with identity matrix)
            if torch.isnan(C_qvim_log).any():
                C_qvim_log = -I_fallback * 10  # Log probabilities approx 0 for matching, very low for non-matching
            if torch.isnan(C_ref_clap_log).any():
                C_ref_clap_log = -I_fallback * 10
            if torch.isnan(C_im_clap_log).any():
                C_im_clap_log = -I_fallback * 10
        
        return C_qvim_log, C_ref_clap_log, C_im_clap_log

    def forward_clap(self, audio):
        """Forward audio through CLAP to get embeddings with extensive error handling"""
        # CLAP expects audio in range [-1, 1] at 16kHz
        batch_size = audio.size(0)
        
        try:
            with torch.no_grad():
                # Manually resample from 32kHz to 16kHz
                import torchaudio
                
                # Debug the audio input
                print(f"DEBUG - Audio shape: {audio.shape}, min: {audio.min()}, max: {audio.max()}")
                
                # First get 2D tensor [batch_size, samples]
                if audio.dim() > 2:
                    audio = audio.squeeze(1)
                
                # Clip the audio to [-1, 1] range
                audio = torch.clamp(audio, -1.0, 1.0)
                
                # Resample from 32kHz to 16kHz
                try:
                    # First pad any very short audio to avoid issues
                    min_length = 16000  # At least 1 second of audio
                    if audio.shape[1] < min_length:
                        padding = min_length - audio.shape[1]
                        audio = F.pad(audio, (0, padding), "constant", 0)
                        print(f"DEBUG - Audio padded to minimum length: {audio.shape}")
                    
                    # Ensure we're dealing with finite values
                    if torch.isnan(audio).any() or torch.isinf(audio).any():
                        audio = torch.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
                        print("DEBUG - Replaced NaN/Inf values in audio input")
                    
                    # Resample with additional checks
                    audio_16k = torchaudio.functional.resample(
                        audio, 
                        orig_freq=32000,  # QVIM sample rate 
                        new_freq=16000    # CLAP required sample rate
                    )
                    
                    # Check for NaN after resampling
                    if torch.isnan(audio_16k).any() or torch.isinf(audio_16k).any():
                        print("WARNING: NaN/Inf values after resampling, replacing with zeros")
                        audio_16k = torch.nan_to_num(audio_16k, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    print(f"DEBUG - Audio resampled to 16kHz: {audio_16k.shape}, min: {audio_16k.min()}, max: {audio_16k.max()}")
                except Exception as e:
                    print(f"ERROR in resampling: {e}")
                    # Create fallback random embeddings
                    return torch.randn(batch_size, 512, device=audio.device)
                
                # Process audio manually with extensive error handling
                try:
                    # Create a list of waveforms as expected by CLAP
                    batch_audio_dict_list = []
                    
                    for i in range(audio_16k.size(0)):
                        # Extract individual waveform
                        waveform = audio_16k[i]
                        
                        # Debug individual waveform
                        if i == 0:
                            print(f"DEBUG - Waveform {i}: shape {waveform.shape}, min {waveform.min()}, max {waveform.max()}")
                        
                        # Make sure the waveform is 1D
                        if waveform.dim() > 1:
                            waveform = waveform.squeeze()
                        
                        # Create audio dict with additional checks
                        audio_dict = {}
                        
                        # Check for NaN or Inf in waveform
                        if torch.isnan(waveform).any() or torch.isinf(waveform).any():
                            print(f"WARNING: NaN or Inf found in waveform {i}, replacing with zeros")
                            waveform = torch.zeros_like(waveform)
                        
                        # Normalize waveform to [-1, 1] range - crucial for CLAP's internal processing
                        if waveform.abs().max() > 0:  # Avoid division by zero
                            waveform = waveform / (waveform.abs().max() + 1e-8)
                        
                        # Add small epsilon to avoid all-zero inputs
                        if waveform.abs().max() < 1e-6:
                            waveform = waveform + torch.randn_like(waveform) * 1e-6
                            print(f"WARNING: Near-zero waveform {i}, adding small noise")
                        
                        # CLAP needs a 1D tensor for audio_data.repeat to work
                        audio_dict["waveform"] = waveform
                        batch_audio_dict_list.append(audio_dict)
                    
                    # Get CLAP embeddings with extensive error handling
                    try:
                        # Add more debug info to diagnose issues
                        for i, audio_dict in enumerate([batch_audio_dict_list[0]]):
                            if i == 0:  # Just check the first item in batch
                                waveform = audio_dict["waveform"]
                                print(f"DEBUG - Waveform stats before CLAP: shape={waveform.shape}, "
                                      f"min={waveform.min():.4f}, max={waveform.max():.4f}, "
                                      f"mean={waveform.mean():.4f}, std={waveform.std():.4f}")
                        
                        with torch.inference_mode():  # More explicit than no_grad in some cases
                            clap_embedding = self.clap_model.model.get_audio_embedding(batch_audio_dict_list)
                        
                        # Gradual handling of NaN/Inf issues
                        if torch.isnan(clap_embedding).any() or torch.isinf(clap_embedding).any():
                            print("WARNING: NaN/Inf found in CLAP embeddings")
                            
                            # First try to recover valid values
                            clap_embedding = torch.nan_to_num(clap_embedding, nan=0.0, posinf=0.0, neginf=0.0)
                            
                            # Check how many rows have all zeros after nan_to_num
                            zero_rows = (clap_embedding.abs().sum(dim=1) < 1e-6).sum().item()
                            if zero_rows > 0:
                                print(f"WARNING: {zero_rows}/{batch_size} CLAP embeddings are all zeros")
                            
                            # For all-zero embeddings, initialize with small random values
                            zero_mask = (clap_embedding.abs().sum(dim=1) < 1e-6).view(-1, 1)
                            if zero_mask.any():
                                random_values = torch.randn_like(clap_embedding) * 0.01
                                clap_embedding = torch.where(zero_mask, random_values, clap_embedding)
                        
                        # Ensure valid embeddings with norm > 0 before normalization
                        norms = torch.norm(clap_embedding, p=2, dim=1, keepdim=True)
                        valid_norms = (norms > 1e-8).float()
                        safe_norms = torch.where(valid_norms > 0, norms, torch.ones_like(norms))
                        
                        # Normalize safely
                        clap_embedding = clap_embedding / safe_norms
                        
                        # Re-compute norm after normalization for debugging
                        print(f"DEBUG - CLAP embedding norm after normalization: {torch.norm(clap_embedding, dim=1).mean():.4f}")
                        
                    except Exception as e:
                        print(f"ERROR in CLAP embedding extraction: {e}")
                        # Create fallback random embeddings
                        clap_embedding = torch.randn(batch_size, 512, device=audio.device)
                    
                except Exception as e:
                    print(f"ERROR in audio processing: {e}")
                    # Create fallback random embeddings
                    clap_embedding = torch.randn(batch_size, 512, device=audio.device)
                
        except Exception as e:
            print(f"CRITICAL ERROR in forward_clap: {e}")
            # Create fallback random embeddings
            clap_embedding = torch.randn(batch_size, 512, device=audio.device)
            
        # Ensure the output is properly normalized with no NaNs
        if torch.isnan(clap_embedding).any():
            print("Final check - NaN still found in CLAP embeddings, using fallback")
            clap_embedding = torch.randn(batch_size, 512, device=audio.device)
            
        clap_embedding = F.normalize(clap_embedding, p=2, dim=1, eps=1e-8)
        return clap_embedding

    def training_step(self, batch, batch_idx):
        # Store current batch for use in lr_scheduler_step
        self.current_batch = batch
        
        self.lr_scheduler_step(batch_idx)

        # Get QVIM embeddings for imitation and reference
        y_imitation, y_reference = self.forward(batch['imitation'], batch['reference'])
        
        # Get CLAP embeddings for reference
        y_clap = self.forward_clap(batch['reference'])
        
        # Calculate similarities with safety checks
        C_qvim_log, C_ref_clap_log, C_im_clap_log = self.compute_safe_similarities(
            y_imitation, y_reference, y_clap
        )
        
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
        # Get QVIM embeddings
        y_imitation, y_reference = self.forward(batch['imitation'], batch['reference'])
        
        # Get CLAP embeddings
        y_clap = self.forward_clap(batch['reference'])
        
        # Calculate similarities with safety checks
        C_qvim_log, C_ref_clap_log, C_im_clap_log = self.compute_safe_similarities(
            y_imitation, y_reference, y_clap
        )
        
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
    
    # Use the existing train function with our custom module
    # Create a factory function to instantiate our module instead of QVIMModule
    def model_factory(config):
        return QVIMCLAPModule(config)
        
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

    # CLAP Configuration
    parser.add_argument('--clap_checkpoint', type=str, 
                        default="",  # Add default path to CLAP checkpoint
                        help="Path to CLAP checkpoint.")
    parser.add_argument('--clap_model', type=str, default="HTSAT-tiny",
                        help="CLAP model architecture to use.")

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