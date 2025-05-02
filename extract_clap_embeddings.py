#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torchaudio
import argparse
import numpy as np
import random
import glob
from pathlib import Path
from typing import Optional, Tuple, Union, List
from audioldm.clap.encoders import CLAPAudioEmbeddingClassifierFreev2
from audioldm.utils import download_checkpoint

def get_clap_embedding(audio_path: str, 
                       audioldm_model: str = "audioldm-m-full",
                       pretrained_clap_path: Optional[str] = None,
                       device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Extract CLAP audio embedding from an audio file.
    
    Args:
        audio_path: Path to the audio file
        audioldm_model: AudioLDM model name to use for CLAP (if pretrained_clap_path not provided)
        pretrained_clap_path: Optional path to standalone CLAP weights
        device: Device to run the model on. If None, uses CUDA if available
    
    Returns:
        torch.Tensor: Audio embedding with shape [1, 512]
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Choose the correct audio model based on AudioLDM model size
    amodel = "HTSAT-tiny"
    if "-m-" in audioldm_model or "-l-" in audioldm_model:
        amodel = "HTSAT-base"
    
    # Initialize the CLAP model
    model = CLAPAudioEmbeddingClassifierFreev2(
        pretrained_path=pretrained_clap_path if pretrained_clap_path else "",
        sampling_rate=16000,  # Input sampling rate for CLAP
        embed_mode="audio",
        amodel=amodel,  # Audio model architecture
    )
    
    # If no pretrained CLAP path is provided, we need to use the AudioLDM checkpoint
    if not pretrained_clap_path:
        print(f"Using CLAP from AudioLDM model: {audioldm_model}")
        print("Note: For complete CLAP functionality, use AudioLDM's built-in CLAP encoder")
    
    model = model.to(device)
    model.eval()
    
    # Load and preprocess audio
    audio, orig_sr = torchaudio.load(audio_path)
    
    # Resample if necessary
    if orig_sr != 16000:
        audio = torchaudio.functional.resample(audio, orig_freq=orig_sr, new_freq=16000)
    
    # Convert to mono if needed
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    # Ensure audio has batch dimension [batch, channels, samples]
    if audio.dim() == 2:
        audio = audio.unsqueeze(0)
    
    # Forward through the model to get embeddings
    with torch.no_grad():
        embedding = model(audio.to(device))
    
    # Remove extra dimensions to get a clean [1, 512] tensor
    if embedding.dim() > 2:
        embedding = embedding.squeeze(1)
    
    return embedding

def find_random_reference_wav() -> str:
    """Find a random reference audio file from the VimSketch dataset"""
    dataset_path = os.path.join("audioldm", "qvim", "data", "Vim_Sketch_Dataset", "references")
    wav_files = glob.glob(os.path.join(dataset_path, "*.wav"))
    
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found in {dataset_path}")
    
    return random.choice(wav_files)

def main():
    parser = argparse.ArgumentParser(description="Extract CLAP audio embeddings from audio files")
    parser.add_argument(
        "--audio_path", 
        type=str, 
        default="random_vimsketch",
        help="Path to audio file or 'random_vimsketch' to use a random file from the dataset"
    )
    parser.add_argument(
        "--audioldm_model", 
        type=str, 
        default="audioldm-m-full",
        choices=["audioldm-s-full", "audioldm-m-full", "audioldm-l-full", 
                "audioldm-s-full-v2", "audioldm-s-text-ft", "audioldm-m-text-ft"],
        help="AudioLDM model to use for CLAP configurations"
    )
    parser.add_argument(
        "--pretrained_clap_path", 
        type=str, 
        required=False,
        default=None,
        help="Optional path to standalone CLAP model weights (if not using AudioLDM's CLAP)"
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        required=False,
        help="Path to save the embedding as a .pt file"
    )
    parser.add_argument(
        "--use_cuda", 
        action="store_true", 
        default=True,
        help="Use CUDA if available"
    )
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Handle the random file case
    if args.audio_path == "random_vimsketch":
        audio_path = find_random_reference_wav()
        print(f"Using random reference file: {audio_path}")
    else:
        audio_path = args.audio_path
    
    # Extract CLAP embedding
    print(f"Extracting CLAP embedding from: {audio_path}")
    embedding = get_clap_embedding(
        audio_path, 
        audioldm_model=args.audioldm_model,
        pretrained_clap_path=args.pretrained_clap_path, 
        device=device
    )
    
    # Print embedding info
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {torch.norm(embedding, dim=1).item()}")
    print(f"First 5 values: {embedding[0, :5].cpu().numpy()}")
    
    # Save embedding if requested
    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        torch.save(embedding, args.save_path)
        print(f"Saved embedding to: {args.save_path}")

if __name__ == "__main__":
    main()