import torch
import os
import numpy as np
import argparse
import librosa
import soundfile as sf  # Needed because save_wave uses sf internally
from tqdm import tqdm
from audioldm.utils import save_wave

# Import our adapter and QVIM utilities
from audioldm.qvim_adapter import (
    QVIMAdapter, 
    load_qvim_model, 
    extract_qvim_embedding
)

# Import AudioLDM components
from audioldm import build_model, seed_everything
from audioldm.pipeline import imitation_to_audio
from audioldm.latent_diffusion.ddim import DDIMSampler

def test_qvim_to_audioldm_pipeline():
    """
    Test the full pipeline from vocal imitation to sound generation.
    This test:
    1. Loads a vocal imitation audio file
    2. Processes it through QVIM to get embeddings
    3. Adapts the embeddings to AudioLDM format
    4. Generates sound using AudioLDM with these embeddings
    
    All operations will run on GPU if available.
    """
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable GPU benchmarking for faster inference
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="Test QVIM to AudioLDM pipeline")
    parser.add_argument("--qvim_checkpoint", type=str, default="audioldm/qvim/models_vimsketch_longer/dulcet-leaf-31/best-loss-checkpoint.ckpt", 
                     help="Path to QVIM model checkpoint")
    parser.add_argument("--audioldm_model", type=str, default="audioldm-s-full", 
                     help="AudioLDM model name or path")
    parser.add_argument("--audio_file", type=str, default=None,
                     help="Path to vocal imitation audio file (optional)")
    parser.add_argument("--output_dir", type=str, default="qvim_pipeline_outputs",
                     help="Directory to save outputs")
    parser.add_argument("--guidance_scale", type=float, default=3.5,
                     help="Classifier-free guidance scale for AudioLDM")
    parser.add_argument("--num_inference_steps", type=int, default=200,
                     help="Number of denoising steps for AudioLDM")
    parser.add_argument("--duration", type=float, default=5.0,
                     help="Duration of generated audio in seconds")
    parser.add_argument("--seed", type=int, default=42,
                     help="Random seed for generation")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed
    seed_everything(args.seed)
    
    print(f"Loading QVIM model from: {args.qvim_checkpoint}")
    qvim_model = load_qvim_model(args.qvim_checkpoint)
    
    # Get the QVIM embedding dimension
    sample_tensor = torch.zeros(1, 32000).to(device)  # 1 second of silence
    with torch.no_grad():
        sample_embedding = qvim_model.forward_imitation(sample_tensor)
    qvim_dim = sample_embedding.shape[1]
    print(f"QVIM embedding dimension: {qvim_dim}")
    
    # Initialize the adapter (AudioLDM uses 512-dim embeddings)
    audioldm_dim = 512
    adapter = QVIMAdapter(qvim_dim, audioldm_dim)
    adapter.to(device)
    
    # Load AudioLDM model
    print(f"Loading AudioLDM model: {args.audioldm_model}")
    latent_diffusion = build_model(model_name=args.audioldm_model)
    
    # Create a DDIMSampler for inference
    sampler = DDIMSampler(latent_diffusion)
    
    # Find audio files
    if args.audio_file and os.path.exists(args.audio_file):
        audio_files = [args.audio_file]
    else:
        # Look for sample audio files in standard locations
        search_dirs = [
            "audioldm/qvim/data/Vim_Sketch_Dataset/vocal_imitations",
            "audioldm/qvim/data/qvim-dev/Queries"
        ]
        
        audio_files = []
        for directory in search_dirs:
            if os.path.exists(directory):
                # Get up to 3 audio files from each directory
                files = []
                for root, _, filenames in os.walk(directory):
                    for filename in filenames:
                        if filename.endswith((".wav", ".mp3", ".flac")):
                            files.append(os.path.join(root, filename))
                            if len(files) >= 3:
                                break
                    if len(files) >= 3:
                        break
                audio_files.extend(files)
        
        if not audio_files:
            print("No audio files found! Using synthetic test data instead.")
            # Create synthetic test data - pure sine waves at different frequencies
            sample_rate = 32000
            duration = 2.0  # 2 seconds
            audio_files = []
            
            # Create directory for synthetic audio
            os.makedirs("synthetic_test_audio", exist_ok=True)
            
            # Create one synthetic example
            freq = 440.0  # A4
            # Use keyword arguments for librosa functions later
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio = 0.5 * np.sin(2 * np.pi * freq * t)
            
            # Add some harmonics to make it more interesting
            audio += 0.3 * np.sin(2 * np.pi * freq * 2 * t)  # Add first harmonic
            
            # Apply envelope
            envelope = np.ones_like(audio)
            attack = int(0.1 * sample_rate)
            release = int(0.3 * sample_rate)
            envelope[:attack] = np.linspace(0, 1, attack)
            envelope[-release:] = np.linspace(1, 0, release)
            audio = audio * envelope
            
            # Save the file
            filename = f"synthetic_test_audio/sine_{int(freq)}hz.wav"
            sf.write(filename, audio, sample_rate)
            audio_files.append(filename)
    
    # Set latent size based on duration
    latent_diffusion.latent_t_size = int(args.duration * 25.6)
    
    # Process each audio file
    print(f"Processing {len(audio_files)} audio files...")
    for idx, audio_path in enumerate(audio_files):
        print(f"\nProcessing file {idx+1}/{len(audio_files)}: {os.path.basename(audio_path)}")
        
        # Use our new pipeline function to generate audio
        print(f"Generating audio from: {audio_path}")
        
        # Generate audio using our pipeline function
        audio_values = imitation_to_audio(
            latent_diffusion=latent_diffusion,
            qvim_model=qvim_model,
            adapter=adapter,
            audio_file_path=audio_path,
            seed=args.seed,
            ddim_steps=args.num_inference_steps,
            duration=args.duration,
            batchsize=1,
            guidance_scale=args.guidance_scale,
            n_candidate_gen_per_text=1,
            audio_type="imitation"
        )
        
        # Load original audio to save as reference
        audio, sr = librosa.load(audio_path, sr=qvim_model.config.sample_rate)
        
        # Use AudioLDM's own save_wave function
        output_basename = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Print diagnostic info about the audio
        print(f"Generated audio shape: {audio_values.shape}")
        
        # Save the generated audio using AudioLDM's function
        filename = f"qvim_generated_{output_basename}"
        save_wave(audio_values, args.output_dir, name=filename)
        
        # Also save the input file for reference (manually since it's in a different format)
        input_audio = np.expand_dims(audio, axis=0)  # Add batch dimension
        input_audio = np.expand_dims(input_audio, axis=1)  # Add channel dimension to match [batch, channel, time]
        save_wave(input_audio, args.output_dir, name=f"input_{output_basename}")
    
    print("\nPipeline test completed successfully!")

if __name__ == "__main__":
    test_qvim_to_audioldm_pipeline()