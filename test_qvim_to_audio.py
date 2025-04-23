#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QVIM-to-Audio Generation Demo

This script demonstrates how to use AudioLDM with QVIM embeddings to generate
sound effects from vocal imitations.

Usage:
  python test_qvim_to_audio.py --audio_file path/to/vocal_imitation.wav
"""

import argparse
import os
import warnings
import torch
import numpy as np
import librosa

# Ignore FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
from audioldm.pipeline import build_model, imitation_to_audio
from audioldm.qvim_adapter import QVIMAdapter, load_qvim_model
from audioldm.utils import save_wave

def parse_args():
    parser = argparse.ArgumentParser(description='Generate audio from vocal imitation')
    parser.add_argument('--audio_file', type=str, required=True,
                        help='Path to vocal imitation audio file')
    parser.add_argument('--qvim_checkpoint', type=str, 
                        default='audioldm/qvim/models_vimsketch_longer/dulcet-leaf-31/best-loss-checkpoint.ckpt',
                        help='Path to QVIM model checkpoint')
    parser.add_argument('--audioldm_model', type=str, default='audioldm-s-full',
                        help='AudioLDM model name')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save generated audio')
    parser.add_argument('--guidance_scale', type=float, default=3.0,
                        help='Classifier-free guidance scale')
    parser.add_argument('--steps', type=int, default=200,
                        help='Number of diffusion steps')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='Duration of generated audio in seconds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.audio_file):
        raise FileNotFoundError(f"Audio file not found: {args.audio_file}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load QVIM model
    print(f"Loading QVIM model from: {args.qvim_checkpoint}")
    qvim_model = load_qvim_model(args.qvim_checkpoint)
    qvim_model = qvim_model.to(device)
    
    # Create QVIM-to-AudioLDM adapter
    qvim_dim = 960  # QVIM embedding dimension
    audioldm_dim = 512  # AudioLDM embedding dimension
    adapter = QVIMAdapter(qvim_dim=qvim_dim, audioldm_dim=audioldm_dim)
    adapter = adapter.to(device)
    
    # Load AudioLDM model
    print(f"Loading AudioLDM model: {args.audioldm_model}")
    latent_diffusion = build_model(model_name=args.audioldm_model)
    latent_diffusion = latent_diffusion.to(device)
    
    # Generate audio from vocal imitation
    print(f"Generating audio from: {args.audio_file}")
    waveform = imitation_to_audio(
        latent_diffusion=latent_diffusion,
        qvim_model=qvim_model,
        adapter=adapter,
        audio_file_path=args.audio_file,
        seed=args.seed,
        ddim_steps=args.steps,
        duration=args.duration,
        batchsize=1,
        guidance_scale=args.guidance_scale,
        n_candidate_gen_per_text=1,
        audio_type="imitation"
    )
    
    # Get output filename base
    output_basename = os.path.splitext(os.path.basename(args.audio_file))[0]
    
    # Print diagnostic info about the audio
    print(f"Generated audio shape: {waveform.shape}")
    
    # Save the generated audio using AudioLDM's function
    filename = f"qvim_generated_{output_basename}"
    save_wave(waveform, args.output_dir, name=filename)
    
    # Also save the input file for reference
    # Load the original audio file
    audio, sr = librosa.load(args.audio_file, sr=32000)  # Use same sample rate as qvim_model
    input_audio = np.expand_dims(audio, axis=0)  # Add batch dimension
    input_audio = np.expand_dims(input_audio, axis=1)  # Add channel dimension to match [batch, channel, time]
    save_wave(input_audio, args.output_dir, name=f"input_{output_basename}")
    
    print(f"Generated audio saved to: {os.path.join(args.output_dir, filename)}")
    print("Input audio saved for reference")

if __name__ == "__main__":
    main()