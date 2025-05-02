# VocalDM: Audio Generation from Vocal Imitation

VocalDM extends AudioLDM to enable generating high-quality audio samples from vocal imitations. This README explains how to use VocalDM with the pretrained QVIM model.

## QVIM Model

Our implementation uses a pretrained Query-by-Vocal-Imitation (QVIM) model that encodes vocal imitations into embedding space. This model is then aligned with AudioLDM's CLAP embedding space to enable vocal-guided audio generation.

### Pretrained Checkpoint

The best performing QVIM model can be found at:
```
audioldm/qvim/models_vimsketch_longer/dulcet-leaf-31/best-loss-checkpoint.ckpt
```

This checkpoint achieved the best validation loss during training and shows strong retrieval performance on test examples.

### Retrieval Examples

We've evaluated the model's retrieval capabilities using single query tests. These examples demonstrate the model's ability to retrieve semantically similar sounds from a database based on vocal imitations.

Example test results can be found at:
```
audioldm/qvim/single_query_tests/dulcet-leaf-31/
```

Each test folder contains:
- An imitation audio file (the vocal input)
- The true match (the actual sound being imitated)
- Top matches retrieved by the model with similarity scores
- A summary.txt file with performance metrics

The model achieves high accuracy in retrieving the correct sounds, demonstrating its ability to capture the semantic content of vocal imitations.

## Integration with AudioLDM

VocalDM connects the QVIM model to AudioLDM through a lightweight adapter network that aligns QVIM embeddings with CLAP's embedding space. This allows AudioLDM to be conditioned on vocal imitations rather than text, while maintaining the quality and diversity of the generated audio.

The trained adapter models can be found in:
```
ckpt/vocaldm/
```

## Usage

There is not yet a checkpoint for VocalDM which produces the desired output.

## Model Architecture

VocalDM combines several components:
1. **QVIM Encoder**: MobileNetV3 network trained on the VimSketch dataset
2. **Adapter Network**: MLP that aligns QVIM embeddings with CLAP audio embeddings
3. **AudioLDM**: Latent diffusion model for high-quality audio generation

Together, these components enable a novel interface for audio generation through vocal imitation.

## Repository Structure

The code in this repository consists of both existing AudioLDM components and newly developed VocalDM code:

- Most files in the root directory are new to VocalDM (with exceptions like `app.py`, `bg.png`, and `setup.py` which existed in the original AudioLDM repo)
- The `audioldm` subfolder contains mostly preexisting AudioLDM code
- The `qvim` subfolder (placed inside `audioldm`) contains the QVIM model implementation
- New components developed specifically for VocalDM include:
  - `audioldm/qvim_adapter.py`: The adapter that bridges QVIM and CLAP embeddings
  - `train_qvim_clap_alignment.py`: Training script for the adapter
  - `train_vocaldm.py`: Training script for the full VocalDM model
  - `test_qvim_adapter.py` and other test scripts
  - Checkpoint management and generation scripts

This repository represents the integration of two separate technologies (AudioLDM and QVIM) into a unified system for vocal-conditioned audio generation.