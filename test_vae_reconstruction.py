#!/usr/bin/env python
import os
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf

# Import AudioLDM components
from vocaldm_utils import load_audioldm_model_with_qvim_cond, waveform_to_mel
from audioldm.qvim_adapter import extract_qvim_embedding, load_qvim_model, QVIMAdapter
from audioldm.qvim.src.qvim_mn_baseline.dataset import VimSketchDataset

def test_vae_reconstruction(audio_file, model_name="audioldm-m-full", save_dir_root="vae_test"):
    """
    Test the VAE reconstruction quality on a given audio file using the same pipeline as train_vocaldm.py.
    
    Args:
        audio_file: Path to input audio file
        model_name: AudioLDM model to use
        save_dir: Directory to save outputs
    """
    print(f"Testing VAE reconstruction on {audio_file}")
    
    # Create save directory, remove extension from audio_file
    save_dir = os.path.join(save_dir_root, audio_file.split("/")[-1].split(".")[0])
    os.makedirs(save_dir, exist_ok=True)
    
    # Load AudioLDM model
    audioldm = load_audioldm_model_with_qvim_cond(model_name)
    print("AudioLDM model loaded successfully")
    
    # Load audio file - using the same sample rate as train_vocaldm.py (32kHz for QVIM, 16kHz for AudioLDM)
    qvim_sample_rate = 32000
    audioldm_sample_rate = 16000
    
    # Load audio using librosa (same as vocaldm_utils.process_audio_file)
    audio, sr = librosa.load(audio_file, sr=qvim_sample_rate)
    
    # Convert to tensor and add batch dimension
    waveform_input = torch.tensor(audio).unsqueeze(0)
    
    print(f"Input waveform shape: {waveform_input.shape}, Sample rate: {qvim_sample_rate}Hz")
    
    # Process the audio to mel spectrogram using the same method as train_vocaldm.py
    # See train_vocaldm.py line 507-512
    mel = waveform_to_mel(
        waveform_input, 
        audioldm_model=audioldm,
        target_shape=(1, 64, 1024),  # AudioLDM's expected mel dimensions for 10s audio
        src_sr=qvim_sample_rate,     # QVIM sample rate (32kHz)
        target_sr=audioldm_sample_rate  # AudioLDM sample rate (16kHz)
    )
    
    print(f"Input mel shape: {mel.shape}")
    
    # Run through VAE encode-decode cycle (exactly as in train_vocaldm.py)
    with torch.no_grad():
        # Move mel to the same device as the model
        device = next(audioldm.parameters()).device
        mel = mel.to(device)
        
        # Encode to latent space - same as train_vocaldm.py line 517-518
        z_latent = audioldm.encode_first_stage(mel)
        z_latent = audioldm.get_first_stage_encoding(z_latent)
        print(f"Latent shape: {z_latent.shape}")
        
        # Decode back to mel spectrogram - same as train_vocaldm.py "decode_first_stage"
        mel_recon = audioldm.decode_first_stage(z_latent)
        print(f"Reconstructed mel shape: {mel_recon.shape}")
        
        # Convert to waveform - same as train_vocaldm.py "mel_spectrogram_to_waveform"
        waveform_recon = audioldm.mel_spectrogram_to_waveform(mel_recon)
        print(f"Reconstructed waveform shape: {waveform_recon.shape}")
    
    # Save original and reconstructed audio
    input_path = os.path.join(save_dir, "input.wav")
    recon_path = os.path.join(save_dir, "reconstructed.wav")
    
    # Save using soundfile (same as done in AudioLDM)
    # Convert to int16 - EXACTLY like in vocoder_infer in hifigan/utilities.py
    print(f"Original waveform shape: {waveform_input.shape}")
    waveform_input = waveform_input.numpy()
    # waveform_input = (waveform_input * 32768).astype(np.int16)
    sf.write(input_path, waveform_input[0], qvim_sample_rate, format='WAV')
    
    # For reconstructed audio, already converted by mel_spectrogram_to_waveform
    sf.write(recon_path, waveform_recon[0][0], audioldm_sample_rate, format='WAV')
    
    # Optional: Plot spectrograms for comparison
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.title("Original Mel Spectrogram")
    plt.imshow(mel[0, 0].cpu().numpy(), aspect='auto', origin='lower')
    
    plt.subplot(2, 1, 2)
    plt.title("Reconstructed Mel Spectrogram")
    plt.imshow(mel_recon[0, 0].cpu().numpy(), aspect='auto', origin='lower')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "spectrogram_comparison.png"))
    
    # Plot the waveforms
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.title("Original Waveform")
    plt.plot(waveform_input[0].numpy())
    
    plt.subplot(2, 1, 2)
    plt.title("Reconstructed Waveform")
    plt.plot(waveform_recon[0])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "waveform_comparison.png"))
    
    print(f"Test complete! Files saved to {save_dir}")
    print(f"Original audio: {input_path}")
    print(f"Reconstructed audio: {recon_path}")

if __name__ == "__main__":
    import sys
    import random
    
    if len(sys.argv) > 1:
        # If file path provided as argument, use it
        audio_file = sys.argv[1]
    else:
        # If no file provided, use a random sample from the VimSketch dataset
        dataset_path = "audioldm/qvim/data/Vim_Sketch_Dataset"
        
        # Load the VimSketch dataset
        full_ds = VimSketchDataset(
            dataset_path,
            sample_rate=32000,
            duration=10.0
        )
        
        # Choose a random sample
        sample_idx = random.randint(0, len(full_ds) - 1)
        sample = full_ds[sample_idx]
        
        # Get the reference audio filename from the sample
        # The dataset returns just the filename, not the full path
        # We need to construct the full path
        filename = sample['reference_filename']
        audio_file = os.path.join(dataset_path, 'references', filename)
        
        print(f"Using random reference audio from VimSketch dataset:")
        print(f"File: {filename}")
        print(f"Full path: {audio_file}")
        print(f"Index: {sample_idx}")
    
    test_vae_reconstruction(audio_file)