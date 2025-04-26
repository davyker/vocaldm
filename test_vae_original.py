#!/usr/bin/env python
import os
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf

# Import AudioLDM components directly
from audioldm.pipeline import build_model
from audioldm.audio import wav_to_fbank, TacotronSTFT, read_wav_file
from audioldm.utils import default_audioldm_config

# Load vocaldm bits
from vocaldm_utils import load_audioldm_model_with_qvim_cond


def test_audioldm_vae_original(audio_file, model_name="audioldm-m-full", save_dir_root="vae_test_original"):
    """
    Test the VAE reconstruction quality using only original AudioLDM code.
    
    Args:
        audio_file: Path to input audio file
        model_name: AudioLDM model to use
        save_dir: Directory to save outputs
    """
    print(f"Testing original AudioLDM VAE reconstruction on {audio_file}")
    
    # Create save directory
    save_dir = os.path.join(save_dir_root, audio_file.split("/")[-1].split(".")[0])
    os.makedirs(save_dir, exist_ok=True)
    
    # Load AudioLDM model
    audioldm = load_audioldm_model_with_qvim_cond(model_name)
    print("AudioLDM model loaded successfully")
    
    # Move to correct device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    audioldm = audioldm.to(device)
    
    # Get default configuration
    config = default_audioldm_config(model_name)
    
    # Create STFT processor as used in original code
    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )
    
    # Target length for 10 seconds of audio at 16kHz
    # In AudioLDM, they use target_length = duration * 102.4
    duration = 10
    target_length = int(duration * 102.4)
    
    # Process audio using original AudioLDM functions
    # This uses wav_to_fbank from audioldm/audio/tools.py
    fbank, log_magnitudes_stft, waveform_input = wav_to_fbank(
        audio_file, 
        target_length=target_length, 
        fn_STFT=fn_STFT
    )
    
    # Add batch and channel dimensions as done in AudioLDM
    # This is the original format used in AudioLDM's pipeline.py
    mel = fbank.unsqueeze(0).unsqueeze(0).to(device)
    
    print(f"Input waveform shape: {waveform_input.shape}")
    print(f"Input mel shape: {mel.shape}")
    
    # Run through VAE encode-decode cycle
    with torch.no_grad():
        # Encode to latent space
        z_posterior = audioldm.encode_first_stage(mel)
        z_latent = audioldm.get_first_stage_encoding(z_posterior)
        print(f"Latent shape: {z_latent.shape}")
        
        # Decode back to mel spectrogram
        mel_recon = audioldm.decode_first_stage(z_latent)
        print(f"Reconstructed mel shape: {mel_recon.shape}")
        
        # Convert to waveform using AudioLDM's function
        waveform_recon = audioldm.mel_spectrogram_to_waveform(mel_recon)
        print(f"Reconstructed waveform shape: {waveform_recon.shape}")
    
    # Save original and reconstructed audio
    input_path = os.path.join(save_dir, "input_original.wav")
    recon_path = os.path.join(save_dir, "reconstructed_original.wav")
    
    # Get original AudioLDM sample rate
    audioldm_sample_rate = config["preprocessing"]["audio"]["sampling_rate"]
    
    # Convert to int16 format
    print(f"Original waveform shape: {waveform_input.shape}")
    waveform_input = waveform_input.numpy()
    # waveform_input = (waveform_input * 32768).astype(np.int16)
    sf.write(input_path, waveform_input, audioldm_sample_rate, format='WAV')
    
    # Convert to int16 format if in float [-1, 1] range
    # waveform_recon = (waveform_recon * 32768).astype(np.int16)
    sf.write(recon_path, waveform_recon[0, 0], audioldm_sample_rate, format='WAV')
    
    # Plot spectrograms for comparison
    plt.figure(figsize=(12, 8))
    
    # Both mel spectrograms have time and frequency axes swapped
    plt.subplot(2, 1, 1)
    plt.title("Original Mel Spectrogram")
    plt.imshow(mel[0, 0].permute(1, 0).cpu().numpy(), aspect='auto', origin='lower')
    
    plt.subplot(2, 1, 2)
    plt.title(f"Reconstructed Mel Spectrogram - MSE difference: {torch.mean((mel_recon[0,0] - mel[0,0]) ** 2).item()}")
    plt.imshow(mel_recon[0, 0].permute(1, 0).cpu().numpy(), aspect='auto', origin='lower')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "spectrogram_comparison_original.png"))
    
    # Plot the waveforms
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.title("Original Waveform")
    plt.plot(waveform_input)
    
    plt.subplot(2, 1, 2)
    plt.title("Reconstructed Waveform")
    plt.plot(waveform_recon[0,0])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "waveform_comparison_original.png"))
    
    print(f"Test complete! Files saved to {save_dir}")
    print(f"Original audio: {input_path}")
    print(f"Reconstructed audio: {recon_path}")

if __name__ == "__main__":
    import sys
    import random

    # Select a random audio file from audioldm/qvim/data/Vim_Sketch_Dataset/references
    audio_dir = "audioldm/qvim/data/Vim_Sketch_Dataset/references"
    audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
    random_audio_file = random.choice(audio_files)
    random_audio_file = os.path.join(audio_dir, random_audio_file)
    print(f"Randomly selected audio file: {random_audio_file}")
    print(f"Using audio file: {random_audio_file}")
    test_audioldm_vae_original(random_audio_file)