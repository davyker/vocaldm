import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import librosa
import numpy as np

class QVIMAdapter(nn.Module):
    """
    Adapter to convert QVIM embeddings to the format expected by AudioLDM.
    """
    def __init__(self, qvim_dim, audioldm_dim):
        """
        Args:
            qvim_dim: Dimension of QVIM embeddings
            audioldm_dim: Dimension expected by AudioLDM's conditioning
        """
        super().__init__()
        
        self.qvim_dim = qvim_dim
        self.audioldm_dim = audioldm_dim
        
        # Multi-layer adapter with normalization
        self.adapter = nn.Sequential(
            nn.Linear(qvim_dim, audioldm_dim * 2),
            nn.LayerNorm(audioldm_dim * 2),
            nn.GELU(),
            nn.Linear(audioldm_dim * 2, audioldm_dim),
            nn.LayerNorm(audioldm_dim)
        )
    
    def forward(self, qvim_embeddings):
        """
        Convert QVIM embeddings to AudioLDM conditioning format.
        
        Args:
            qvim_embeddings: Tensor of shape [batch_size, qvim_dim]
                
        Returns:
            Adapted embeddings compatible with AudioLDM's FiLM conditioning: [batch_size, audioldm_dim]
        """
        # Apply adapter
        adapted = self.adapter(qvim_embeddings)
        
        # L2 normalize like CLAP does
        adapted = F.normalize(adapted, p=2, dim=-1)
        
        # For FiLM conditioning, we don't need the extra sequence dimension
        # The model expects [batch_size, audioldm_dim]
        if adapted.dim() == 3:
            adapted = adapted.squeeze(1)  # Remove sequence dimension if present
            
        return adapted
    
    def get_unconditional_embedding(self, batch_size=1, device="cuda"):
        """
        Generate unconditional embeddings for classifier-free guidance.
        Similar to how CLAP generates embeddings for empty strings.
        
        Args:
            batch_size: Number of unconditional embeddings to generate
            device: Device to create tensor on
            
        Returns:
            Unconditional embeddings: [batch_size, audioldm_dim]
        """
        with torch.no_grad():  # Disable gradient computation
            # Create a zero embedding that matches the expected dimensions for FiLM conditioning
            uncond_embedding = torch.zeros(batch_size, self.audioldm_dim, device=device)
            
            # Apply same normalization as conditional embeddings
            uncond_embedding = F.normalize(uncond_embedding, p=2, dim=-1)
        
        return uncond_embedding

def extract_qvim_embedding(audio_tensor, qvim_model, audio_type="imitation"):
   """
   Extract QVIM embedding from raw audio.
   
   Args:
       audio_tensor: Raw audio tensor of shape [batch_size, audio_length]
       qvim_model: Loaded QVIM model (QVIMModule instance)
       audio_type: Type of audio - "imitation" or "sound"
       
   Returns:
       QVIM embedding tensor of shape [batch_size, embedding_dim]
   """
   # Move to the correct device
   device = next(qvim_model.parameters()).device
   audio_tensor = audio_tensor.to(device)
   
   # Extract embeddings based on audio type
   with torch.no_grad():
       if audio_type == "imitation":
           # Use the forward_imitation method which handles mel conversion internally
           embedding = qvim_model.forward_imitation(audio_tensor)
       elif audio_type == "sound":
           # Use the forward_reference method which handles mel conversion internally
           embedding = qvim_model.forward_reference(audio_tensor)
       else:
           raise ValueError(f"Unknown audio_type: {audio_type}. Must be 'imitation' or 'sound'")
   
   return embedding

def load_audio_batch(audio_paths, sample_rate):
   """
   Load and preprocess a batch of audio files.
   
   Args:
       audio_paths: List of paths to audio files
       sample_rate: Target sample rate
       
   Returns:
       Audio tensor of shape [batch_size, audio_length]
   """
   batch_audio = []
   max_length = 0
   
   # First pass to find max length
   for path in audio_paths:
       audio, sr = librosa.load(path, sr=sample_rate)
       max_length = max(max_length, len(audio))
       batch_audio.append(audio)
   
   # Pad all audio to the same length
   padded_batch = []
   for audio in batch_audio:
       padded = np.pad(audio, (0, max_length - len(audio)))
       padded_batch.append(padded)
   
   # Convert to tensor
   audio_tensor = torch.tensor(np.stack(padded_batch))
   
   return audio_tensor

def prepare_vocim_conditioning(audio_tensor, qvim_model, adapter, audio_type="imitation", unconditional_guidance_scale=3.0):
   """
   Prepare conditioning from audio for AudioLDM.
   
   Args:
       audio_tensor: Raw audio tensor of shape [batch_size, audio_length]
       qvim_model: Loaded QVIM model
       adapter: QVIMAdapter instance
       audio_type: Type of audio - "imitation" or "sound"
       unconditional_guidance_scale: Scale for classifier-free guidance
       
   Returns:
       Tuple of (adapted_embeddings, uncond_embeddings) to use directly with sample_log
   """
   # Extract embeddings from the audio
   embeddings = extract_qvim_embedding(audio_tensor, qvim_model, audio_type=audio_type)
   
   # Transform the embeddings using the adapter
   adapted_embeddings = adapter(embeddings)
   
   # Prepare conditioning based on guidance scale
   if unconditional_guidance_scale > 1.0:
       # Get unconditional embeddings
       uncond_embeddings = adapter.get_unconditional_embedding(
           batch_size=adapted_embeddings.shape[0],
           device=adapted_embeddings.device
       )
       return adapted_embeddings, uncond_embeddings
   else:
       return adapted_embeddings, None

def load_qvim_model(checkpoint_path):
   """
   Load the QVIM model from a checkpoint.
   
   Args:
       checkpoint_path: Path to the QVIM model checkpoint
                       
   Returns:
       Loaded QVIM model
   """
   # Load the saved checkpoint
   checkpoint = torch.load(checkpoint_path, map_location='cpu')
   
   # Import QVIMModule from the right location
   from audioldm.qvim.src.qvim_mn_baseline.ex_qvim import QVIMModule
   
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
           self.initial_tau = 0.07  # Will be overridden by the checkpoint
           self.tau_trainable = False
   
   # Create a model with default configuration
   model = QVIMModule(Config())
   
   # Load the weights
   model.load_state_dict(checkpoint['state_dict'])
   
   # Set to evaluation mode
   model.eval()
   
   # Move to GPU if available
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)
   
   return model

def process_audio_file(audio_path, qvim_model, adapter, audio_type="imitation", unconditional_guidance_scale=3.0):
   """
   Process a single audio file for conditioning AudioLDM.
   
   Args:
       audio_path: Path to audio file
       qvim_model: Loaded QVIM model
       adapter: QVIMAdapter instance
       audio_type: Type of audio - "imitation" or "sound"
       unconditional_guidance_scale: Scale for classifier-free guidance
       
   Returns:
       Dictionary with conditioning for AudioLDM
   """
   # Load the audio file
   audio, sr = librosa.load(audio_path, sr=qvim_model.config.sample_rate)
   audio_tensor = torch.tensor(audio).unsqueeze(0)  # Add batch dimension
   
   # Process through the conditioning pipeline
   return prepare_vocim_conditioning(
       audio_tensor, 
       qvim_model, 
       adapter, 
       audio_type=audio_type,
       unconditional_guidance_scale=unconditional_guidance_scale
   )