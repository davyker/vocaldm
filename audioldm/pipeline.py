import os

import argparse
import yaml
import torch
from torch import autocast
from tqdm import tqdm, trange

from audioldm import LatentDiffusion, seed_everything
from audioldm.utils import default_audioldm_config, get_duration, get_bit_depth, get_metadata, download_checkpoint
from audioldm.audio import wav_to_fbank, TacotronSTFT, read_wav_file
from audioldm.latent_diffusion.ddim import DDIMSampler
from einops import repeat
import os

def make_batch_for_text_to_audio(text, waveform=None, fbank=None, batchsize=1):
    text = [text] * batchsize
    if batchsize < 1:
        print("Warning: Batchsize must be at least 1. Batchsize is set to .")
    
    if(fbank is None):
        fbank = torch.zeros((batchsize, 1024, 64))  # Not used, here to keep the code format
    else:
        fbank = torch.FloatTensor(fbank)
        fbank = fbank.expand(batchsize, 1024, 64)
        assert fbank.size(0) == batchsize
        
    stft = torch.zeros((batchsize, 1024, 512))  # Not used

    if(waveform is None):
        waveform = torch.zeros((batchsize, 160000))  # Not used
    else:
        waveform = torch.FloatTensor(waveform)
        waveform = waveform.expand(batchsize, -1)
        assert waveform.size(0) == batchsize
        
    fname = [""] * batchsize  # Not used
    
    batch = (
        fbank,
        stft,
        None,
        fname,
        waveform,
        text,
    )
    return batch

def make_batch_for_imitation_to_audio(qvim_embeddings, waveform=None, fbank=None, batchsize=1):
    """
    Create a batch for imitation-to-audio generation.
    
    Args:
        qvim_embeddings: QVIM embeddings from vocal imitation
        waveform: Optional waveform data (not used for conditioning)
        fbank: Optional filterbank data (not used for conditioning)
        batchsize: Batch size for generation
        
    Returns:
        Batch tuple compatible with the AudioLDM pipeline
    """
    # Expand QVIM embeddings if needed (already in tensor format)
    if batchsize > 1 and qvim_embeddings.size(0) == 1:
        qvim_embeddings = qvim_embeddings.expand(batchsize, -1)
    
    if batchsize < 1:
        print("Warning: Batchsize must be at least 1. Batchsize is set to 1.")
    
    # Determine device from qvim_embeddings
    device = qvim_embeddings.device if isinstance(qvim_embeddings, torch.Tensor) else torch.device('cpu')
    
    # Create placeholder data for unused fields
    if fbank is None:
        fbank = torch.zeros((batchsize, 1024, 64), device=device)  # Not used, here to keep the code format
    else:
        # Check if it's already a tensor
        if isinstance(fbank, torch.Tensor):
            # Ensure it's a float tensor and keep it on its original device
            fbank = fbank.float()
        else:
            # Convert to float tensor if it's not already a tensor
            fbank = torch.FloatTensor(fbank).to(device)
            
        # Expand if needed
        if fbank.size(0) != batchsize or fbank.dim() < 3:
            fbank = fbank.expand(batchsize, 1024, 64)
            
        assert fbank.size(0) == batchsize
        
    stft = torch.zeros((batchsize, 1024, 512), device=device)  # Not used

    if waveform is None:
        waveform = torch.zeros((batchsize, 160000), device=device)  # Not used
    else:
        # Check if it's already a tensor
        if isinstance(waveform, torch.Tensor):
            # Ensure it's a float tensor and keep it on its original device
            waveform = waveform.float()
        else:
            # Convert to float tensor if it's not already a tensor
            waveform = torch.FloatTensor(waveform).to(device)
            
        # Expand if needed
        if waveform.size(0) != batchsize:
            waveform = waveform.expand(batchsize, -1)
            
        assert waveform.size(0) == batchsize
        
    fname = [""] * batchsize  # Not used
    
    # Put the QVIM embeddings in the text position of the batch
    # This allows us to reuse existing AudioLDM infrastructure
    # The conditioning key will determine how this is processed
    
    # Create the batch tuple in the format expected by AudioLDM
    batch = (
        fbank,
        stft,
        None,
        fname,
        waveform,
        qvim_embeddings,  # QVIM embeddings in place of text
    )
    return batch

def round_up_duration(duration):
    return int(round(duration/2.5) + 1) * 2.5

def build_model(
    ckpt_path=None,
    config=None,
    model_name="audioldm-s-full"
):
    print("Load AudioLDM: %s", model_name)
    
    if(ckpt_path is None):
        ckpt_path = get_metadata()[model_name]["path"]
    
    if(not os.path.exists(ckpt_path)):
        download_checkpoint(model_name)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if config is not None:
        assert type(config) is str
        config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    else:
        config = default_audioldm_config(model_name)

    # Use text as condition instead of using waveform during training
    config["model"]["params"]["device"] = device
    config["model"]["params"]["cond_stage_key"] = "text"

    # No normalization here
    latent_diffusion = LatentDiffusion(**config["model"]["params"])

    resume_from_checkpoint = ckpt_path

    checkpoint = torch.load(resume_from_checkpoint, map_location=device)
    '''Original. Here is a bug that, an unexpected key "cond_stage_model.model.text_branch.embeddings.position_ids" exists in the checkpoint file. '''
    # latent_diffusion.load_state_dict(checkpoint["state_dict"])
    '''2023.10.17 Fix the bug by setting the paramer "strict" as "False" to ignore the unexpected key. '''
    latent_diffusion.load_state_dict(checkpoint["state_dict"], strict=False)

    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.to(device)

    latent_diffusion.cond_stage_model.embed_mode = "text"
    return latent_diffusion

def duration_to_latent_t_size(duration):
    return int(duration * 25.6)

def set_cond_audio(latent_diffusion):
    latent_diffusion.cond_stage_key = "waveform"
    latent_diffusion.cond_stage_model.embed_mode="audio"
    return latent_diffusion

def set_cond_text(latent_diffusion):
    latent_diffusion.cond_stage_key = "text"
    latent_diffusion.cond_stage_model.embed_mode="text"
    return latent_diffusion
    
def set_cond_qvim(latent_diffusion):
    """
    Set the conditioning key to use QVIM embeddings.
    This creates a direct path for QVIM embeddings to be used as conditioning.
    
    Args:
        latent_diffusion: The AudioLDM model
        
    Returns:
        The modified AudioLDM model
    """
    # Set the same key as text conditioning, since we're using 
    # the same FiLM conditioning mechanism, just with different embeddings
    latent_diffusion.cond_stage_key = "text"
    # Set embed mode to bypass since we're providing our own embeddings
    latent_diffusion.cond_stage_model.embed_mode = "bypass" 
    return latent_diffusion
    
def text_to_audio(
    latent_diffusion,
    text,
    original_audio_file_path = None,
    seed=42,
    ddim_steps=200,
    duration=10,
    batchsize=1,
    guidance_scale=2.5,
    n_candidate_gen_per_text=3,
    config=None,
):
    seed_everything(int(seed))
    waveform = None
    if(original_audio_file_path is not None):
        waveform = read_wav_file(original_audio_file_path, int(duration * 102.4) * 160)
        
    batch = make_batch_for_text_to_audio(text, waveform=waveform, batchsize=batchsize)

    latent_diffusion.latent_t_size = duration_to_latent_t_size(duration)
    
    if(waveform is not None):
        print("Generate audio that has similar content as %s" % original_audio_file_path)
        latent_diffusion = set_cond_audio(latent_diffusion)
    else:
        print("Generate audio using text %s" % text)
        latent_diffusion = set_cond_text(latent_diffusion)
        
    with torch.no_grad():
        waveform = latent_diffusion.generate_sample(
            [batch],
            unconditional_guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            n_candidate_gen_per_text=n_candidate_gen_per_text,
            duration=duration,
        )
    return waveform


def imitation_to_audio(
    latent_diffusion,
    qvim_model,
    adapter,
    audio_file_path,
    seed=42,
    ddim_steps=200,
    duration=10,
    batchsize=1,
    guidance_scale=2.5,
    n_candidate_gen_per_text=3,
    audio_type="imitation",
):
    """
    Generates audio from a vocal imitation using AudioLDM conditioned on QVIM embeddings.
    
    Args:
        latent_diffusion: AudioLDM latent diffusion model instance
        qvim_model: QVIM model instance for extracting embeddings
        adapter: QVIMAdapter instance to convert QVIM embeddings to AudioLDM format
        audio_file_path: Path to the audio file containing vocal imitation
        seed: Random seed for generation
        ddim_steps: Number of denoising steps
        duration: Duration of generated audio in seconds
        batchsize: Batch size for generation
        guidance_scale: Scale for classifier-free guidance
        n_candidate_gen_per_text: Number of candidates to generate per imitation
        audio_type: Type of audio input ("imitation" or "sound")
        
    Returns:
        Generated waveform(s)
    """
    from audioldm.qvim_adapter import extract_qvim_embedding, load_audio_batch
    
    seed_everything(int(seed))
    
    # Load and process the audio file
    import librosa
    sample_rate = qvim_model.config.sample_rate if hasattr(qvim_model.config, "sample_rate") else 32000
    audio, _ = librosa.load(audio_file_path, sr=sample_rate)
    audio_tensor = torch.tensor(audio).unsqueeze(0).to(next(qvim_model.parameters()).device)
    
    # Extract QVIM embeddings
    print(f"Extracting QVIM embeddings from {audio_file_path}")
    qvim_embeddings = extract_qvim_embedding(audio_tensor, qvim_model, audio_type=audio_type)
    
    # Set audio duration
    latent_diffusion.latent_t_size = duration_to_latent_t_size(duration)
    
    # Configure model to use bypass mode for conditioning
    latent_diffusion = set_cond_qvim(latent_diffusion)
    
    # Adapt QVIM embeddings to AudioLDM format (960-dim to 512-dim)
    adapted_embeddings = adapter(qvim_embeddings)
    
    # Duplicate for multiple candidates if needed
    if n_candidate_gen_per_text > 1:
        adapted_embeddings = torch.cat([adapted_embeddings] * n_candidate_gen_per_text, dim=0)
        actual_batchsize = batchsize * n_candidate_gen_per_text
    else:
        actual_batchsize = batchsize
    
    # For classifier-free guidance, we need unconditional embeddings
    if guidance_scale > 1.0:
        unconditional_embedding = adapter.get_unconditional_embedding(
            batch_size=actual_batchsize,
            device=adapted_embeddings.device
        )
    else:
        unconditional_embedding = None
    
    print(f"Generating audio with QVIM conditioning (guidance scale: {guidance_scale})")
    # For generation (test_vocaldm.py), we can still use no_grad, but for training it will be handled differently
    # in train_vocaldm.py by using training_ema_scope
    with torch.no_grad():
        with latent_diffusion.ema_scope("Generate sample with QVIM..."):
            # Set up shape for sampling
            shape = (latent_diffusion.channels, latent_diffusion.latent_t_size, latent_diffusion.latent_f_size)
            
            # Create DDIM sampler (try to use training-compatible version first)
            try:
                # Try to use the training-compatible sampler first for consistency
                from audioldm.latent_diffusion.ddim_for_training import DDIMSamplerForTraining
                sampler = DDIMSamplerForTraining(latent_diffusion)
            except ImportError:
                # Fall back to original sampler if not available
                from audioldm.latent_diffusion.ddim import DDIMSampler
                sampler = DDIMSampler(latent_diffusion)
                
            sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=1.0, verbose=False)
            
            # Run sampling with properly formatted conditioning
            samples, _ = sampler.sample(
                S=ddim_steps,
                batch_size=actual_batchsize,
                shape=shape,
                conditioning=adapted_embeddings,  # Direct tensor input, not a dictionary
                unconditional_guidance_scale=guidance_scale,
                unconditional_conditioning=unconditional_embedding,
                verbose=True
            )
            
            # Check for extreme values
            if torch.max(torch.abs(samples)) > 1e2:
                samples = torch.clip(samples, min=-10, max=10)
                
            # Decode samples to mel spectrograms
            mel = latent_diffusion.decode_first_stage(samples)
            
            # Convert mel spectrograms to waveforms  
            waveform = latent_diffusion.mel_spectrogram_to_waveform(mel)
            
            # If we generated multiple candidates, select the best one
            if n_candidate_gen_per_text > 1 and waveform.shape[0] > batchsize:
                # We'll simplify by just taking the first candidate for each input
                # In a real scenario, you might want to use a quality metric
                selected_indices = list(range(0, waveform.shape[0], n_candidate_gen_per_text))
                waveform = waveform[selected_indices]
            
            # Print shape for debugging
            print(f"Audio waveform shape in pipeline: {waveform.shape}")
    
    return waveform

def style_transfer(
    latent_diffusion,
    text,
    original_audio_file_path,
    transfer_strength,
    seed=42,
    duration=10,
    batchsize=1,
    guidance_scale=2.5,
    ddim_steps=200,
    config=None,
):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    assert original_audio_file_path is not None, "You need to provide the original audio file path"
    
    audio_file_duration = get_duration(original_audio_file_path)
    
    assert get_bit_depth(original_audio_file_path) == 16, "The bit depth of the original audio file %s must be 16" % original_audio_file_path
    
    # if(duration > 20):
    #     print("Warning: The duration of the audio file %s must be less than 20 seconds. Longer duration will result in Nan in model output (we are still debugging that); Automatically set duration to 20 seconds")
    #     duration = 20
    
    if(duration > audio_file_duration):
        print("Warning: Duration you specified %s-seconds must equal or smaller than the audio file duration %ss" % (duration, audio_file_duration))
        duration = round_up_duration(audio_file_duration)
        print("Set new duration as %s-seconds" % duration)

    # duration = round_up_duration(duration)
    
    latent_diffusion = set_cond_text(latent_diffusion)

    if config is not None:
        assert type(config) is str
        config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    else:
        config = default_audioldm_config()

    seed_everything(int(seed))
    # latent_diffusion.latent_t_size = duration_to_latent_t_size(duration)
    latent_diffusion.cond_stage_model.embed_mode = "text"

    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )

    mel, _, _ = wav_to_fbank(
        original_audio_file_path, target_length=int(duration * 102.4), fn_STFT=fn_STFT
    )
    mel = mel.unsqueeze(0).unsqueeze(0).to(device)
    mel = repeat(mel, "1 ... -> b ...", b=batchsize)
    init_latent = latent_diffusion.get_first_stage_encoding(
        latent_diffusion.encode_first_stage(mel)
    )  # move to latent space, encode and sample
    if(torch.max(torch.abs(init_latent)) > 1e2):
        init_latent = torch.clip(init_latent, min=-10, max=10)
    sampler = DDIMSampler(latent_diffusion)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=1.0, verbose=False)

    t_enc = int(transfer_strength * ddim_steps)
    prompts = text

    with torch.no_grad():
        with autocast("cuda"):
            with latent_diffusion.ema_scope():
                uc = None
                if guidance_scale != 1.0:
                    uc = latent_diffusion.cond_stage_model.get_unconditional_condition(
                        batchsize
                    )

                c = latent_diffusion.get_learned_conditioning([prompts] * batchsize)
                z_enc = sampler.stochastic_encode(
                    init_latent, torch.tensor([t_enc] * batchsize).to(device)
                )
                samples = sampler.decode(
                    z_enc,
                    c,
                    t_enc,
                    unconditional_guidance_scale=guidance_scale,
                    unconditional_conditioning=uc,
                )
                # x_samples = latent_diffusion.decode_first_stage(samples) # Will result in Nan in output
                # print(torch.sum(torch.isnan(samples)))
                x_samples = latent_diffusion.decode_first_stage(samples)
                # print(x_samples)
                x_samples = latent_diffusion.decode_first_stage(samples[:,:,:-3,:])
                # print(x_samples)
                waveform = latent_diffusion.first_stage_model.decode_to_waveform(
                    x_samples
                )

    return waveform

def super_resolution_and_inpainting(
    latent_diffusion,
    text,
    original_audio_file_path = None,
    seed=42,
    ddim_steps=200,
    duration=None,
    batchsize=1,
    guidance_scale=2.5,
    n_candidate_gen_per_text=3,
    time_mask_ratio_start_and_end=(0.10, 0.15), # regenerate the 10% to 15% of the time steps in the spectrogram
    # time_mask_ratio_start_and_end=(1.0, 1.0), # no inpainting
    # freq_mask_ratio_start_and_end=(0.75, 1.0), # regenerate the higher 75% to 100% mel bins
    freq_mask_ratio_start_and_end=(1.0, 1.0), # no super-resolution
    config=None,
):
    seed_everything(int(seed))
    if config is not None:
        assert type(config) is str
        config = yaml.load(open(config, "r"), Loader=yaml.FullLoader)
    else:
        config = default_audioldm_config()
    fn_STFT = TacotronSTFT(
        config["preprocessing"]["stft"]["filter_length"],
        config["preprocessing"]["stft"]["hop_length"],
        config["preprocessing"]["stft"]["win_length"],
        config["preprocessing"]["mel"]["n_mel_channels"],
        config["preprocessing"]["audio"]["sampling_rate"],
        config["preprocessing"]["mel"]["mel_fmin"],
        config["preprocessing"]["mel"]["mel_fmax"],
    )
    
    # waveform = read_wav_file(original_audio_file_path, None)
    mel, _, _ = wav_to_fbank(
        original_audio_file_path, target_length=int(duration * 102.4), fn_STFT=fn_STFT
    )
    
    batch = make_batch_for_text_to_audio(text, fbank=mel[None,...], batchsize=batchsize)
        
    # latent_diffusion.latent_t_size = duration_to_latent_t_size(duration)
    latent_diffusion = set_cond_text(latent_diffusion)
        
    with torch.no_grad():
        waveform = latent_diffusion.generate_sample_masked(
            [batch],
            unconditional_guidance_scale=guidance_scale,
            ddim_steps=ddim_steps,
            n_candidate_gen_per_text=n_candidate_gen_per_text,
            duration=duration,
            time_mask_ratio_start_and_end=time_mask_ratio_start_and_end,
            freq_mask_ratio_start_and_end=freq_mask_ratio_start_and_end
        )
    return waveform
