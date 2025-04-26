#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VocalDM Test Script

This script tests the AudioLDM model with vocal imitation conditioning.
It demonstrates the full pipeline:
1. Load an audio file (vocal imitation)
2. Extract QVIM embeddings
3. Adapt the embeddings to AudioLDM's format
4. Generate audio using AudioLDM conditioned on these embeddings

Usage:
  python test_vocaldm.py --audio_file path/to/vocal_imitation.wav --output_dir outputs
"""

import argparse
import os
import torch
import torch.nn as nn
import librosa
import numpy as np
from tqdm import tqdm

# Import AudioLDM components
from audioldm import build_model, seed_everything
from audioldm.utils import save_wave
from audioldm.pipeline import set_cond_qvim, duration_to_latent_t_size
from audioldm.qvim_adapter import QVIMAdapter, load_qvim_model, extract_qvim_embedding

def parse_args():
    parser = argparse.ArgumentParser(description='Test AudioLDM with QVIM')
    parser.add_argument('--audio_file', type=str, required=True,
                      help='Path to vocal imitation audio file')
    parser.add_argument('--qvim_checkpoint', type=str,
                      default='audioldm/qvim/models_vimsketch_longer/dulcet-leaf-31/best-loss-checkpoint.ckpt',
                      help='Path to QVIM model checkpoint')
    parser.add_argument('--adapter_checkpoint', type=str, default=None,
                      help='Path to trained QVIM adapter checkpoint')
    parser.add_argument('--audioldm_model', type=str, default='audioldm-s-full',
                      help='AudioLDM model name')
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Directory to save generated audio')
    parser.add_argument('--duration', type=float, default=10.0,
                      help='Duration of generated audio in seconds (AudioLDM is optimized for 10s)')
    parser.add_argument('--guidance_scale', type=float, default=3.0,
                      help='Classifier-free guidance scale')
    parser.add_argument('--ddim_steps', type=int, default=100,
                      help='Number of denoising steps')
    parser.add_argument('--ddim_eta', type=float, default=1.0,
                      help='DDIM eta parameter (0.0 = deterministic, 1.0 = stochastic)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode with more verbose output')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    seed_everything(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device (cuda if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load QVIM model
    print(f"Loading QVIM model from: {args.qvim_checkpoint}")
    qvim_model = load_qvim_model(args.qvim_checkpoint)
    qvim_model = qvim_model.to(device)
    
    # Load AudioLDM model
    print(f"Loading AudioLDM model: {args.audioldm_model}")
    model = build_model(model_name=args.audioldm_model)
    model = model.to(device)
    
    # Set up the QVIM-to-AudioLDM adapter
    print("Setting up QVIM adapter")
    qvim_dim = 960  # QVIM embedding dimension 
    audioldm_dim = 512  # AudioLDM embedding dimension
    adapter = QVIMAdapter(qvim_dim=qvim_dim, audioldm_dim=audioldm_dim)
    
    # Load adapter checkpoint and trained film parameters if provided
    if args.adapter_checkpoint is not None:
        print(f"Loading checkpoint from: {args.adapter_checkpoint}")
        checkpoint = torch.load(args.adapter_checkpoint, map_location='cpu')
        
        # Check what's in the checkpoint for debugging
        print(f"Checkpoint contains keys: {checkpoint.keys()}")
        
        # Load the adapter parameters
        if 'adapter' in checkpoint:
            # This is the new format with separate adapter and state_dict keys
            print("Found 'adapter' key - using new checkpoint format")
            adapter.load_state_dict(checkpoint['adapter'])
            
            # If state_dict is present, load FiLM parameters too
            if 'state_dict' in checkpoint:
                print("Loading FiLM conditioning parameters...")
                film_params_loaded = 0
                
                # Load the FiLM parameters into the model
                for key, value in checkpoint['state_dict'].items():
                    if not key.startswith('adapter.') and any(x in key for x in ['cond_emb', 'film', 'label_emb', 'time_embed']):
                        try:
                            # Navigate the model hierarchy to set the parameter
                            components = key.split('.')
                            current = model
                            for component in components[:-1]:
                                if not hasattr(current, component):
                                    print(f"Warning: Component '{component}' not found in model when processing key '{key}'")
                                    break
                                current = getattr(current, component)
                            else:  # This executes if the for loop completes without break
                                if not hasattr(current, components[-1]):
                                    print(f"Warning: Final attribute '{components[-1]}' not found in model when processing key '{key}'")
                                else:
                                    setattr(current, components[-1], nn.Parameter(value))
                                    film_params_loaded += 1
                                    print(f"Loaded FiLM parameter: {key}")
                        except Exception as e:
                            print(f"Error loading FiLM parameter {key}: {e}")
                
                if film_params_loaded > 0:
                    print(f"Successfully loaded {film_params_loaded} FiLM conditioning parameters")
                else:
                    print("No FiLM conditioning parameters were loaded")
        elif 'model_state_dict' in checkpoint:
            # Old format with only model_state_dict key
            print("Found 'model_state_dict' key - using legacy format")
            adapter.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            # Lightning checkpoint with single state_dict key
            print("Found 'state_dict' key - extracting adapter parameters")
            adapter_state_dict = {
                k.replace('adapter.', ''): v for k, v in checkpoint['state_dict'].items()
                if k.startswith('adapter.')
            }
            adapter.load_state_dict(adapter_state_dict)
            
            # Look for FiLM parameters in state_dict
            print("Looking for FiLM conditioning parameters in the checkpoint...")
            film_params_loaded = 0
            
            for key in checkpoint['state_dict'].keys():
                if not key.startswith('adapter.') and any(x in key for x in ['cond_emb', 'film', 'label_emb', 'time_embed']):
                    try:
                        # Navigate the model hierarchy to set the parameter
                        components = key.split('.')
                        current = model
                        for component in components[:-1]:
                            if not hasattr(current, component):
                                print(f"Warning: Component '{component}' not found in model when processing key '{key}'")
                                break
                            current = getattr(current, component)
                        else:  # This executes if the for loop completes without break
                            if not hasattr(current, components[-1]):
                                print(f"Warning: Final attribute '{components[-1]}' not found in model when processing key '{key}'")
                            else:
                                setattr(current, components[-1], nn.Parameter(checkpoint['state_dict'][key]))
                                film_params_loaded += 1
                                print(f"Loaded FiLM parameter: {key}")
                    except Exception as e:
                        print(f"Error loading FiLM parameter {key}: {e}")
            
            if film_params_loaded > 0:
                print(f"Successfully loaded {film_params_loaded} FiLM conditioning parameters")
            else:
                print("No FiLM conditioning parameters were loaded")
        else:
            # Assume it's a direct adapter state_dict
            print("Assuming checkpoint is a direct adapter state_dict")
            adapter.load_state_dict(checkpoint)
        
        print("Checkpoint loaded successfully")
            
    adapter = adapter.to(device)
    
    # Configure AudioLDM to use bypass mode for external embeddings
    model = set_cond_qvim(model)
    
    # Set audio duration
    model.latent_t_size = duration_to_latent_t_size(args.duration)
    print(f"Latent temporal size: {model.latent_t_size}")
    
    # Load and process input audio
    print(f"Loading audio file: {args.audio_file}")
    sample_rate = qvim_model.config.sample_rate if hasattr(qvim_model.config, "sample_rate") else 32000
    audio, _ = librosa.load(args.audio_file, sr=sample_rate)
    audio_tensor = torch.tensor(audio).unsqueeze(0).to(device)
    
    # Extract QVIM embeddings
    print("Extracting QVIM embeddings")
    with torch.no_grad():
        qvim_embeddings = extract_qvim_embedding(audio_tensor, qvim_model, audio_type="imitation")
        
    # Print embedding shape for debugging
    if args.debug:
        print(f"QVIM embedding shape: {qvim_embeddings.shape}")
        print(f"QVIM embedding type: {qvim_embeddings.dtype}")
    
    # Adapt embeddings to AudioLDM format
    print("Adapting embeddings to AudioLDM format")
    with torch.no_grad():
        adapted_embeddings = adapter(qvim_embeddings)
    
    if args.debug:
        print(f"Adapted embedding shape: {adapted_embeddings.shape}")
        print(f"Adapted embedding type: {adapted_embeddings.dtype}")
        print(f"Adapted embedding device: {adapted_embeddings.device}")
    
    # Generate unconditional embeddings for classifier-free guidance
    if args.guidance_scale > 1.0:
        print(f"Creating unconditional embeddings (guidance scale: {args.guidance_scale})")
        unconditional_embedding = adapter.get_unconditional_embedding(
            batch_size=1,
            device=device
        )
    else:
        unconditional_embedding = None
    
    # Set up DDIM sampler (using our training-compatible version for consistency)
    try:
        # Try to use the training-compatible sampler first
        from audioldm.latent_diffusion.ddim_for_training import DDIMSamplerForTraining
        print("Setting up DDIM sampler (training-compatible version)")
        sampler = DDIMSamplerForTraining(model)
    except ImportError:
        # Fall back to original sampler if not available
        from audioldm.latent_diffusion.ddim import DDIMSampler
        print("Setting up DDIM sampler (standard version)")
        sampler = DDIMSampler(model)
    
    sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=args.ddim_eta, verbose=True)
    
    # Set up shape for sampling
    shape = (model.channels, model.latent_t_size, model.latent_f_size)
    if args.debug:
        print(f"Generation shape: {shape}")
    
    # Generate audio
    print(f"Generating audio (steps: {args.ddim_steps})")
    with torch.no_grad():
        with model.ema_scope("Generating audio"):
            # Run sampling with properly formatted conditioning
            samples, _ = sampler.sample(
                S=args.ddim_steps,
                batch_size=1,
                shape=shape,
                conditioning=adapted_embeddings,  # Direct tensor input
                unconditional_guidance_scale=args.guidance_scale,
                unconditional_conditioning=unconditional_embedding,
                eta=args.ddim_eta,  # Explicitly pass eta to avoid the default override
                verbose=True
            )
            
            # Check for extreme values
            if torch.max(torch.abs(samples)) > 1e2:
                print("Clipping extreme values in latent")
                samples = torch.clip(samples, min=-10, max=10)
                
            # Decode samples to mel spectrograms
            print("Decoding latents to mel spectrograms")
            mel = model.decode_first_stage(samples)
            
            # Convert mel spectrograms to waveforms
            print("Converting mel spectrograms to waveforms")  
            waveform = model.mel_spectrogram_to_waveform(mel)
            
            print(f"Generated audio shape: {waveform.shape}")
    
    # Create organized folder structure
    # Get the vocal imitation name (filename without extension)
    imitation_name = os.path.splitext(os.path.basename(args.audio_file))[0]
    
    # Create a folder for this imitation if it doesn't exist
    imitation_folder = os.path.join(args.output_dir, imitation_name)
    os.makedirs(imitation_folder, exist_ok=True)
    
    # Extract checkpoint identifier (last 2 parts, replacing '/' with '__')
    if args.adapter_checkpoint:
        # Extract the last two parts of the path
        checkpoint_parts = args.adapter_checkpoint.split('/')[-2:]
        # Replace slashes with double underscores
        checkpoint_id = '__'.join(checkpoint_parts)
    else:
        checkpoint_id = "no_adapter"
    
    # Create a folder for this checkpoint if it doesn't exist
    output_folder = os.path.join(imitation_folder, checkpoint_id)
    os.makedirs(output_folder, exist_ok=True)
    
    # Create a metadata file with generation parameters
    import datetime
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    timestamp_for_filename = now.strftime("%Y%m%d_%H%M%S")
    
    metadata_path = os.path.join(output_folder, f"generation_info_{timestamp_for_filename}.txt")
    with open(metadata_path, "w") as f:
        f.write(f"Generation timestamp: {timestamp}\n")
        f.write(f"Vocal imitation: {args.audio_file}\n")
        f.write(f"Adapter checkpoint: {args.adapter_checkpoint}\n")
        f.write(f"AudioLDM model: {args.audioldm_model}\n")
        f.write(f"Duration: {args.duration} seconds\n")
        f.write(f"Guidance scale: {args.guidance_scale}\n")
        f.write(f"DDIM steps: {args.ddim_steps}\n")
        f.write(f"DDIM eta: {args.ddim_eta} ({args.ddim_eta == 0.0 and 'deterministic' or args.ddim_eta == 1.0 and 'stochastic' or 'partially stochastic'})\n")
        f.write(f"Random seed: {args.seed}\n")
    
    # Save the generated audio
    print(f"Saving generated audio to: {output_folder}")
    save_wave(waveform, output_folder, name=f"vocaldm_output_{timestamp_for_filename}")
    
    # Save input audio for reference
    input_audio = np.expand_dims(audio, axis=0)  # Add batch dimension
    input_audio = np.expand_dims(input_audio, axis=1)  # Add channel dimension
    save_wave(input_audio, output_folder, name=f"input_reference_{timestamp_for_filename}")
    
    print("Generation complete!")

if __name__ == "__main__":
    main()