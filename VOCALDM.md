# VocalDM: Voice Imitation-to-Audio Generation

VocalDM extends AudioLDM to generate realistic audio from vocal imitations. 
This document describes how to train and use the model.

## Overview

VocalDM combines two key components:
1. A QVIM (Query-by-Vocal-Imitation) model that encodes vocal imitations into embedding vectors
2. AudioLDM's latent diffusion architecture that generates audio conditioned on these embeddings

The integration uses a trained adapter to convert QVIM embeddings to a format compatible with AudioLDM's conditioning mechanism.

## Training

VocalDM requires fine-tuning the cross-attention layers that connect conditioning information to AudioLDM's diffusion process. These layers need to be retrained to understand vocal imitation embeddings from QVIM.

### Training Script

The training is handled by `train_vocaldm.py`, which fine-tunes only the:
- Cross-attention layers in AudioLDM's diffusion model (key/value projections)
- QVIM adapter module

```bash
python train_vocaldm.py \
  --audioldm_checkpoint ckpt/audioldm-m-full.ckpt \
  --qvim_checkpoint audioldm/qvim/models/breezy-resonance-18/best-mrr-checkpoint.ckpt \
  --project vocaldm \
  --run_name vocaldm_run1 \
  --batch_size 8 \
  --adapter_lr 1e-4 \
  --attention_lr 1e-5 \
  --max_epochs 30 \
  --checkpoint_dir ckpt/vocaldm
```

### Key Parameters

- `--audioldm_checkpoint`: Path to pretrained AudioLDM model
- `--qvim_checkpoint`: Path to pretrained QVIM model
- `--adapter_lr`: Learning rate for the adapter module
- `--attention_lr`: Learning rate for cross-attention layers
- `--dataset_path`: Path to dataset directory (default: audioldm/qvim/data)
- `--duration`: Duration of audio clips in seconds (default: 5.0)
- `--batch_size`: Number of samples per batch (default: 8)
- `--guidance_scale`: Classifier-free guidance scale (default: 3.0)

For a complete list of parameters, run:
```bash
python train_vocaldm.py --help
```

### Training Process

The training uses the VimSketch dataset, which contains pairs of vocal imitations and corresponding reference sounds. The training process:

1. Extracts embeddings from vocal imitations using QVIM
2. Transforms them using the adapter
3. Uses these embeddings to condition AudioLDM
4. Fine-tunes cross-attention layers to understand vocal imitation embeddings
5. Tries to reconstruct the reference sound from the vocal imitation embedding

## Inference

Once trained, you can use VocalDM to generate sounds from vocal imitations:

```bash
python test_vocaldm.py \
  --audioldm_checkpoint ckpt/audioldm-m-full.ckpt \
  --qvim_checkpoint audioldm/qvim/models/breezy-resonance-18/best-mrr-checkpoint.ckpt \
  --adapter_checkpoint ckpt/vocaldm/qvim_adapter.pt \
  --input_audio path/to/vocal_imitation.wav \
  --output_dir output/generated \
  --guidance_scale 3.0 \
  --ddim_steps 50
```

The script will:
1. Load the pretrained models and fine-tuned adapter
2. Extract QVIM embeddings from your vocal imitation
3. Generate a sound that matches the vocal imitation
4. Save both the input and generated audio to the output directory

You can also provide a directory of audio files to process multiple imitations:
```bash
python test_vocaldm.py \
  --audioldm_checkpoint ckpt/audioldm-m-full.ckpt \
  --qvim_checkpoint audioldm/qvim/models/breezy-resonance-18/best-mrr-checkpoint.ckpt \
  --adapter_checkpoint ckpt/vocaldm/qvim_adapter.pt \
  --input_audio path/to/imitations_directory \
  --output_dir output/generated
```

## Customization

- Adjust `--guidance_scale` to control how much the generation follows the conditioning (higher values follow more closely)
- Increase `--ddim_steps` for higher quality but slower generation
- Use different QVIM checkpoints for different vocal imitation capabilities