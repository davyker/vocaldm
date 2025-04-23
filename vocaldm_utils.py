#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import gc
import torch
import numpy as np
import librosa
import time
import torch
import torchaudio.transforms as T

from audioldm.ldm import LatentDiffusion
from audioldm.pipeline import make_batch_for_imitation_to_audio, set_cond_qvim
from audioldm.qvim_adapter import QVIMAdapter, extract_qvim_embedding, load_qvim_model
from audioldm.audio.tools import _pad_spec  # Import AudioLDM's padding function

# Map of model names to Hugging Face repo IDs
AUDIOLDM_REPO_MAP = {
    "audioldm-s-full": "cvssp/audioldm-s-full",
    "audioldm-m-full": "cvssp/audioldm-m-full",  # Recommended by authors
    "audioldm-l-full": "cvssp/audioldm-l-full",
    "audioldm-s-full-v2": "cvssp/audioldm-s-full-v2",
    "audioldm-s-text-ft": "cvssp/audioldm-s-text-ft",
    "audioldm-m-text-ft": "cvssp/audioldm-m-text-ft"
}

VERBOSE = False

def load_audioldm_model(model_name_or_path, device=None):
    """
    Load an AudioLDM model from a model name or local checkpoint path
    
    Args:
        model_name_or_path: Model name from AUDIOLDM_REPO_MAP or path to checkpoint
        device: Device to load the model on (cuda, cpu, etc.)
        
    Returns:
        AudioLDM model configured for QVIM
    """
    from audioldm.pipeline import build_model
    
    if os.path.exists(model_name_or_path):
        # It's a local file path
        print(f"Loading AudioLDM from checkpoint: {model_name_or_path}")
        audioldm = build_model(ckpt_path=model_name_or_path)
    elif model_name_or_path in AUDIOLDM_REPO_MAP:
        # It's a known model name
        print(f"Loading AudioLDM model: {model_name_or_path}")
        audioldm = build_model(model_name=model_name_or_path)
    else:
        # Try to use as model name or fall back to default
        try:
            print(f"Attempting to load model: {model_name_or_path}")
            audioldm = build_model(model_name=model_name_or_path)
        except Exception as e:
            print(f"Failed to load model, falling back to default. Error: {e}")
            default_model = "audioldm-m-full"
            print(f"Loading default AudioLDM model: {default_model}")
            audioldm = build_model(model_name=default_model)
    
    # Configure for QVIM conditioning
    audioldm = set_cond_qvim(audioldm)
    
    return audioldm

def setup_qvim_and_adapter(qvim_checkpoint, qvim_dim=960, audioldm_dim=512, adapter_checkpoint=None):
    """
    Load QVIM model and set up adapter
    
    Args:
        qvim_checkpoint: Path to QVIM checkpoint
        qvim_dim: QVIM embedding dimension
        audioldm_dim: AudioLDM embedding dimension
        adapter_checkpoint: Optional path to pre-trained adapter checkpoint
        
    Returns:
        Tuple of (qvim_model, adapter)
    """
    print(f"Loading QVIM model from {qvim_checkpoint}")
    qvim_model = load_qvim_model(qvim_checkpoint)
    qvim_model.eval()  # Set to evaluation mode
    
    # Initialize adapter
    print(f"Initializing QVIM adapter: QVIM dim={qvim_dim}, AudioLDM dim={audioldm_dim}")
    adapter = QVIMAdapter(qvim_dim, audioldm_dim)
    
    # Load adapter weights if provided
    if adapter_checkpoint and os.path.exists(adapter_checkpoint):
        print(f"Loading adapter weights from {adapter_checkpoint}")
        adapter.load_state_dict(torch.load(adapter_checkpoint))
    
    adapter.eval()  # Set to evaluation mode
    
    return qvim_model, adapter

def process_audio_file(file_path, qvim_model, sample_rate=None):
    """
    Load and preprocess an audio file for QVIM
    
    Args:
        file_path: Path to audio file
        qvim_model: QVIM model for determining sample rate
        sample_rate: Optional sample rate override
        
    Returns:
        Tuple of (audio tensor, sample rate)
    """
    # Determine sample rate
    if sample_rate is None:
        sample_rate = qvim_model.config.sample_rate
    
    # Load audio
    audio, sr = librosa.load(file_path, sr=sample_rate)
    
    # Resample if needed
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    
    # Convert to tensor and add batch dimension
    audio_tensor = torch.tensor(audio).unsqueeze(0)
    
    return audio_tensor, sample_rate

def cleanup_resources():
    """Clean up GPU resources"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Resources cleaned up")

def waveform_to_mel(waveform, audioldm_model=None, target_shape=(1, 64, 64), src_sr=32000, target_sr=16000):
    """
    Convert raw waveform to mel spectrogram in the format expected by AudioLDM
    
    Args:
        waveform: Raw audio waveform tensor [batch_size, samples]
        audioldm_model: AudioLDM model (optional, for using its mel converter)
        target_shape: Target shape for the mel spectrogram (channels, height, width)
        src_sr: Source sample rate of the waveform (default: 32000 Hz for QVIM)
        target_sr: Target sample rate for AudioLDM (default: 16000 Hz)
        
    Returns:
        Mel spectrogram tensor [batch_size, channels, height, width]
    """
    batch_size = waveform.shape[0]
    device = waveform.device
    
    # Use the provided target sample rate
    sample_rate = target_sr
    n_fft = 1024
    hop_length = 160
    win_length = 1024
    n_mels = 64
    fmin = 0
    fmax = 8000
    
    # Get device from waveform
    device = waveform.device
    
    print(f"Processing audio batch with shape {waveform.shape} on device {device}") if VERBOSE else None
    
    mel_specs = []
    for i in range(batch_size):
        # Process one audio at a time
        audio_i = waveform[i]  # [samples]
        
        # Convert to mono if needed
        if audio_i.dim() > 1:
            audio_i = torch.mean(audio_i, dim=0)
        
        # Resample if needed
        if src_sr != target_sr:
            print(f"Resampling audio from {src_sr}Hz to {target_sr}Hz") if VERBOSE else None
            
            # Resample on CPU to avoid CUDA issues with torchaudio
            audio_i_cpu = audio_i.cpu()
            resampler = T.Resample(
                orig_freq=src_sr,
                new_freq=target_sr
            )
            
            # Resample and move back to original device
            audio_i = resampler(audio_i_cpu).to(device)
            print(f"Resampled audio shape: {audio_i.shape}") if VERBOSE else None
        
        # Normalize audio to [-1, 1] range
        if torch.abs(audio_i).max() > 1.0:
            audio_i = audio_i / torch.abs(audio_i).max()
        
        # Use torchaudio's MelSpectrogram directly (much more GPU-friendly)
        mel_transform = T.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            f_min=fmin,
            f_max=fmax,
            n_mels=n_mels,
            power=2.0
        ).to(device)
        
        # Generate mel spectrogram - torchaudio expects [channels, time]
        audio_i = audio_i.unsqueeze(0)  # Add channel dim: [1, samples]
        mel_spec = mel_transform(audio_i)  # [1, n_mels, time]
        
        # Convert to log scale like AudioLDM's preprocessing
        mel_spec = torch.log(mel_spec + 1e-5)
        
        # Match AudioLDM's normalization
        mel_spec = (mel_spec + 4.5) / 5.0  # Fast normalization
        
        # Print mel_spec shape for debugging
        print(f"Generated mel spec for item {i}: shape={mel_spec.shape}") if VERBOSE else None
        
        # Add batch dimension -> [1, 1, n_mels, time]
        mel_output = mel_spec.unsqueeze(0)
        mel_specs.append(mel_output)
    
    # Stack all specs in batch
    mel = torch.cat(mel_specs, dim=0)
    
    # Ensure mel has the right shape and use AudioLDM's padding approach
    print(f"Before padding, mel shape: {mel.shape}, target: {target_shape}") if VERBOSE else None
    
    # The default target shape for AudioLDM should be (1, 64, 1024) for 10-second audio
    channels, height, width = target_shape
    
    # AudioLDM expects 1024 time frames for 10-second audio
    # If the target shape has width != 1024 and it's 64, we should adjust it
    if width != 1024 and width == 64:
        print("Warning: Target width 64 is incorrect for AudioLDM. Adjusting to 1024.")
        target_shape = (channels, height, 1024)
    
    # Our mel is in format [batch, 1, n_mels, time]
    # Just check time dimension and use AudioLDM's pad_spec directly
    if mel.shape[3] != target_shape[2]:
        # Just use _pad_spec to adjust the time dimension
        print(f"Padding/cutting time dimension from {mel.shape[3]} to {target_shape[2]} frames") if VERBOSE else None
        
        # Apply padding/cutting - output keeps same dimensions except for time
        if mel.shape[3] < target_shape[2]:
            # Need to pad
            p = target_shape[2] - mel.shape[3]
            padder = torch.nn.ZeroPad2d((0, p, 0, 0))  # (left, right, top, bottom)
            mel = padder(mel)
        else:
            # Need to cut
            mel = mel[:, :, :, :target_shape[2]]
    
    print(f"Final mel spectrogram shape: {mel.shape}") if VERBOSE else None
    return mel

def make_vocaldm_batch(qvim_embeddings, waveform=None, batchsize=1):
    """
    Create a batch for vocal imitation to audio generation
    
    Args:
        qvim_embeddings: QVIM embeddings
        waveform: Optional waveform for supervised training
        batchsize: Batch size
        
    Returns:
        Batch format expected by AudioLDM with QVIM conditioning
    """
    return make_batch_for_imitation_to_audio(
        qvim_embeddings=qvim_embeddings,
        waveform=waveform,
        batchsize=batchsize
    )

def select_cross_attention_params(model):
    """
    Select cross-attention key and value projection parameters for training
    
    Args:
        model: AudioLDM model
        
    Returns:
        List of parameter tensors for cross-attention key/value projections
    """
    from audioldm.latent_diffusion.attention import CrossAttention
    
    attention_params = []
    for name, module in model.named_modules():
        if isinstance(module, CrossAttention) and name.endswith('attn2'):
            attention_params.extend(list(module.to_k.parameters()))
            attention_params.extend(list(module.to_v.parameters()))
    
    return attention_params