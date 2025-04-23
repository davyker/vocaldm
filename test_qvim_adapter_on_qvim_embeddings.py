import torch
import numpy as np
import os
import argparse
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm

def test_qvim_adapter_with_real_embeddings():
    """
    Test the QVIMAdapter with real QVIM embeddings from vocal imitations.
    This test:
    1. Loads the QVIM model
    2. Processes sample vocal imitations to get embeddings
    3. Passes the embeddings through the adapter
    4. Verifies output format and visualizes embedding distributions
    """
    parser = argparse.ArgumentParser(description="Test QVIM adapter with real embeddings")
    parser.add_argument("--qvim_checkpoint", type=str, default="audioldm/qvim/models_vimsketch_longer/dulcet-leaf-31/best-loss-checkpoint.ckpt", 
                      help="Path to QVIM model checkpoint")
    parser.add_argument("--test_audio", type=str, default=None,
                      help="Path to a specific test audio file (optional)")
    parser.add_argument("--audioldm_dim", type=int, default=512,
                      help="Dimension expected by AudioLDM (default: 512)")
    args = parser.parse_args()
    
    # Import the QVIM modules (do this inside function to avoid import errors if run directly)
    from audioldm.qvim_adapter import QVIMAdapter, load_qvim_model, extract_qvim_embedding
    
    # Load the QVIM model
    print(f"Loading QVIM model from: {args.qvim_checkpoint}")
    qvim_model = load_qvim_model(args.qvim_checkpoint)
    
    # Get the QVIM embedding dimension (should be 1024 for MobileNetV3)
    sample_tensor = torch.zeros(1, 32000).to(qvim_model.device)  # 1 second of silence
    with torch.no_grad():
        sample_embedding = qvim_model.forward_imitation(sample_tensor)
    qvim_dim = sample_embedding.shape[1]
    print(f"QVIM embedding dimension: {qvim_dim}")
    
    # Initialize the adapter
    adapter = QVIMAdapter(qvim_dim, args.audioldm_dim)
    adapter.to(qvim_model.device)
    
    # Process audio files
    if args.test_audio:
        # Process a single specified audio file
        audio_files = [args.test_audio]
    else:
        # Look for sample audio files in standard locations
        search_dirs = [
            "audioldm/qvim/data/Vim_Sketch_Dataset/vocal_imitations",
            "audioldm/qvim/data/qvim-dev/Queries"
        ]
        
        audio_files = []
        for directory in search_dirs:
            if os.path.exists(directory):
                # Get up to 5 audio files from each directory
                files = []
                for root, _, filenames in os.walk(directory):
                    for filename in filenames:
                        if filename.endswith((".wav", ".mp3", ".flac")):
                            files.append(os.path.join(root, filename))
                            if len(files) >= 5:
                                break
                    if len(files) >= 5:
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
            
            for i, freq in enumerate([261.63, 440.0, 880.0, 1760.0]):  # C4, A4, A5, A6
                t = np.linspace(0, duration, int(sample_rate * duration), False)
                audio = 0.5 * np.sin(2 * np.pi * freq * t)
                
                # Add some harmonics to make it more interesting
                if i % 2 == 0:
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
    
    # Process each audio file
    all_embeddings = []
    all_adapted = []
    filenames = []
    
    print(f"Processing {len(audio_files)} audio files...")
    for audio_path in tqdm(audio_files):
        # Load audio
        audio, sr = librosa.load(audio_path, sr=qvim_model.config.sample_rate)
        audio_tensor = torch.tensor(audio).unsqueeze(0).to(qvim_model.device)
        
        # Extract QVIM embedding
        embedding = extract_qvim_embedding(audio_tensor, qvim_model, audio_type="imitation")
        
        # Adapt the embedding
        adapted = adapter(embedding)
        
        # Store results - detach from computation graph before converting to numpy
        all_embeddings.append(embedding.detach().cpu().numpy())
        all_adapted.append(adapted.detach().cpu().numpy())
        filenames.append(os.path.basename(audio_path))
    
    # Process the results
    print("\nResults:")
    for i, filename in enumerate(filenames):
        print(f"File: {filename}")
        print(f"  QVIM embedding shape: {all_embeddings[i].shape}")
        print(f"  Adapted embedding shape: {all_adapted[i].shape}")
        
        # Verify L2 normalization
        norm = np.linalg.norm(all_adapted[i][0, 0])
        print(f"  Adapted embedding norm: {norm:.6f}")
    
    # Create a conditioning dictionary (as would be used in AudioLDM)
    if all_adapted:
        # Convert back to tensors for demonstration
        adapted_tensor = torch.tensor(all_adapted[0]).to(qvim_model.device)
        uncond_embedding = adapter.get_unconditional_embedding(
            batch_size=adapted_tensor.shape[0],
            device=adapted_tensor.device
        )
        
        cond = {
            "c_crossattn": [adapted_tensor, uncond_embedding]
        }
        
        print("\nConditioning dictionary structure:")
        print(f"Keys: {list(cond.keys())}")
        print(f"c_crossattn list length: {len(cond['c_crossattn'])}")
        print(f"Shapes: {[t.shape for t in cond['c_crossattn']]}")
        
        # Visualize embeddings
        if len(all_embeddings) > 1:
            try:
                plt.figure(figsize=(15, 6))
                
                # Original QVIM embeddings - take first 20 dimensions for visualization
                plt.subplot(1, 2, 1)
                data = np.array([emb[0, :20] for emb in all_embeddings])
                plt.imshow(data, aspect='auto', cmap='viridis')
                plt.title('QVIM Embeddings (first 20 dims)')
                plt.xlabel('Embedding Dimension')
                plt.ylabel('Audio Sample')
                plt.colorbar()
                
                # Adapted embeddings
                plt.subplot(1, 2, 2)
                data = np.array([emb[0, 0, :20] for emb in all_adapted])
                plt.imshow(data, aspect='auto', cmap='viridis')
                plt.title('Adapted Embeddings (first 20 dims)')
                plt.xlabel('Embedding Dimension')
                plt.ylabel('Audio Sample')
                plt.colorbar()
                
                plt.tight_layout()
                plt.savefig('qvim_adapter_embeddings.png')
                print("\nEmbedding visualization saved to 'qvim_adapter_embeddings.png'")
            except Exception as e:
                print(f"Could not create visualization: {e}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_qvim_adapter_with_real_embeddings()