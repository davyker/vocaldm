#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified VocalDM training script

This script trains ONLY the adapter layers to convert
QVIM embeddings to AudioLDM conditioning format.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb
import time

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import adapter from audioldm
from audioldm.qvim_adapter import QVIMAdapter
from audioldm.qvim.src.qvim_mn_baseline.ex_qvim import QVIMModule

class SimpleVocalDMTrainer:
    """
    Simple trainer for VocalDM that only trains the adapter between QVIM and AudioLDM
    without involving the diffusion process.
    """
    def __init__(self, args):
        self.args = args
        self.target_dim = args.audioldm_dim
        self.setup_models()
        self.setup_optimizer()
        self.setup_logger()
        
    def setup_models(self):
        # Load the QVIM model (frozen)
        print(f"Loading QVIM model from {self.args.qvim_checkpoint}")
        checkpoint = torch.load(self.args.qvim_checkpoint, map_location='cpu')
        
        # Create a basic config with default values based on the QVIM paper
        class Config:
            def __init__(self):
                # Audio settings
                self.pretrained_name = "mn10_as"
                self.n_mels = 128
                self.sample_rate = 32000
                self.window_size = 800
                self.hop_size = 320
                self.n_fft = 1024
                self.freqm = 8
                self.timem = 300
                self.fmin = 0
                self.fmax = None
                self.fmin_aug_range = 10
                self.fmax_aug_range = 2000
                
                # Model settings
                self.initial_tau = 0.07
                self.tau_trainable = False
        
        self.qvim_model = QVIMModule(Config())
        self.qvim_model.load_state_dict(checkpoint['state_dict'])
        self.qvim_model.to(device)
        self.qvim_model.eval()  # Set to eval mode, we won't train this
        
        # Freeze all QVIM parameters
        for param in self.qvim_model.parameters():
            param.requires_grad = False
        
        # Create the adapter (trainable)
        self.adapter = QVIMAdapter(
            qvim_dim=self.args.qvim_dim, 
            audioldm_dim=self.args.audioldm_dim
        )
        self.adapter.to(device)
        self.adapter.train()  # Set to train mode
        
        # Create fixed target embeddings that mimic CLAP/AudioLDM embeddings
        # We'll use these to train the adapter in a supervised way
        # In a real setup, these would be from CLAP/AudioLDM for specific sounds
        
        # Generate some fixed random "target embeddings" representing different sounds
        np.random.seed(42)  # For reproducibility
        num_targets = 20
        
        # Create embeddings with reasonable norms
        self.target_embeddings = []
        for i in range(num_targets):
            vec = np.random.randn(1, self.target_dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)  # Normalize to unit length
            self.target_embeddings.append(torch.tensor(vec, device=device))
        
        # Labels for each target embedding (just for logging)
        self.target_labels = [
            "dog bark", "door bell", "footsteps", "glass breaking", "gun shot",
            "keyboard typing", "phone ringing", "rain", "speech", "traffic",
            "water flowing", "wind blowing", "bird chirping", "car horn", "clock ticking",
            "cough", "crying baby", "door knock", "laughing", "music"
        ]
    
    def setup_optimizer(self):
        # Setup optimizer for the adapter
        self.optimizer = torch.optim.AdamW(
            self.adapter.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.args.epochs,
            eta_min=self.args.min_lr
        )
    
    def setup_logger(self):
        # Setup wandb for logging
        if self.args.log_to_wandb:
            wandb.init(
                project=self.args.wandb_project,
                name=self.args.wandb_run_name or f"vocaldm-simple-{time.strftime('%Y%m%d-%H%M%S')}",
                config=vars(self.args)
            )
    
    def extract_qvim_embedding(self, audio_tensor):
        """Extract QVIM embedding from audio"""
        with torch.no_grad():
            # Use forward_imitation method for vocal imitations
            embedding = self.qvim_model.forward_imitation(audio_tensor)
        return embedding
    
    def train_step(self, audio_tensor, target_idx):
        """
        Simple training step:
        1. Extract QVIM embedding
        2. Pass through adapter
        3. Compute loss against a target AudioLDM embedding
        4. Backpropagate and update weights
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Extract QVIM embedding
        with torch.no_grad():
            qvim_embedding = self.extract_qvim_embedding(audio_tensor)
        
        # Pass through adapter
        adapted_embedding = self.adapter(qvim_embedding)
        
        # Get target embedding
        target_embedding = self.target_embeddings[target_idx]
        
        # Compute cosine similarity loss (we want to maximize similarity)
        # Both embeddings should already be L2 normalized
        cosine_sim = F.cosine_similarity(
            adapted_embedding, 
            target_embedding.expand_as(adapted_embedding),
            dim=-1
        )
        loss = -cosine_sim.mean()  # Negative because we want to maximize similarity
        
        # Compute MSE loss as an alternative
        mse_loss = F.mse_loss(adapted_embedding, target_embedding.expand_as(adapted_embedding))
        
        # Combine losses
        combined_loss = loss + mse_loss
        
        # Backpropagate and update weights
        combined_loss.backward()
        self.optimizer.step()
        
        return {
            "loss": combined_loss.item(),
            "cosine_loss": loss.item(),
            "mse_loss": mse_loss.item(),
            "cosine_sim": -loss.item()  # Convert back to similarity
        }
    
    def log_metrics(self, metrics, step):
        """Log metrics to wandb"""
        if self.args.log_to_wandb:
            wandb.log(metrics, step=step)
        
        # Print metrics to console
        if step % self.args.print_every == 0:
            metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
            print(f"Step {step}: {metrics_str}")
    
    def save_checkpoint(self, path, step=None):
        """Save adapter checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            "adapter_state_dict": self.adapter.state_dict(),
            "qvim_dim": self.args.qvim_dim,
            "audioldm_dim": self.args.audioldm_dim,
            "step": step,
            "args": vars(self.args)
        }
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
        # Also log to wandb
        if self.args.log_to_wandb:
            wandb.save(path)
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        # Simple synthetic data
        # Create some white noise tensors to represent different vocalizations
        num_samples = 32000  # 1 second at 32kHz
        
        total_steps = self.args.steps
        best_loss = float('inf')
        
        for step in tqdm(range(total_steps), desc="Training"):
            # Generate random audio tensor
            batch_size = self.args.batch_size
            audio_tensor = torch.randn(batch_size, num_samples, device=device)
            
            # Pick random target embeddings for this batch
            target_indices = torch.randint(
                0, len(self.target_embeddings), 
                (batch_size,), 
                device=device
            )
            
            # Train step
            metrics = self.train_step(audio_tensor, target_indices[0])  # Use first index for simplicity
            
            # Log metrics
            self.log_metrics(metrics, step)
            
            # Update scheduler
            if (step + 1) % self.args.batch_size == 0:
                self.scheduler.step()
                
            # Save checkpoint
            if (step + 1) % self.args.save_every == 0:
                self.save_checkpoint(
                    os.path.join(self.args.output_dir, f"adapter_step_{step+1}.pt"),
                    step=step
                )
                
            # Save best model
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                self.save_checkpoint(
                    os.path.join(self.args.output_dir, "adapter_best.pt"),
                    step=step
                )
        
        # Save final model
        self.save_checkpoint(
            os.path.join(self.args.output_dir, "adapter_final.pt"),
            step=total_steps
        )
        
        print("Training completed!")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple VocalDM Adapter Training")
    
    # Basic parameters
    parser.add_argument("--qvim_checkpoint", type=str, 
                       default="audioldm/qvim/models_vimsketch_longer/dulcet-leaf-31/best-mrr-checkpoint.ckpt",
                       help="Path to QVIM model checkpoint")
    parser.add_argument("--qvim_dim", type=int, default=960,
                       help="Dimension of QVIM embeddings")
    parser.add_argument("--audioldm_dim", type=int, default=512,
                       help="Dimension of AudioLDM embeddings")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                       help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6,
                       help="Weight decay")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("--steps", type=int, default=10000,
                       help="Total number of training steps")
    
    # Logging and saving
    parser.add_argument("--output_dir", type=str, default="ckpt/vocaldm",
                       help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=1000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--print_every", type=int, default=100,
                       help="Print metrics every N steps")
    parser.add_argument("--log_to_wandb", action="store_true",
                       help="Whether to log to wandb")
    parser.add_argument("--wandb_project", type=str, default="vocaldm",
                       help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Wandb run name")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = SimpleVocalDMTrainer(args)
    
    # Train
    trainer.train()
    
    # Cleanup
    if args.log_to_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()