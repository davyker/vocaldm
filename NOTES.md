# Captain's Log: VocalDM Development Process

This document chronicles the development process, challenges, and insights gained during the creation of VocalDM, a system for generating audio from vocal imitations.

## Project Overview

The goal was to combine two existing models:
1. **AudioLDM**: A latent diffusion model that generates audio clips from text using CLAP text embeddings (standard setup with CNN VAE and U-net)
2. **QVIM model**: Uses MobileNetV3 CNN pre-trained on AudioSet for vocal imitation retrieval

By replacing AudioLDM's text conditioning with QVIM conditioning, I aimed to create a system that could generate sounds from vocal imitations rather than text descriptions.

## Development Process

### Phase 1: QVIM Training and Evaluation

I first trained the QVIM model on its own to ensure it would effectively encode vocal imitations. The results were promising:
- The ground truth files were usually in the top 5 retrieved results
- Other retrieved sounds were semantically similar to the expected sound

This validated that the QVIM model could effectively capture the semantic content of vocal imitations.

### Phase 2: Adapter Development

I developed an adapter to bridge the dimensional gap between models:
- CLAP embeddings: 512-dimensional (normalized)
- QVIM embeddings: 960-dimensional (normalized)

The adapter consisted of a two-layer MLP:
```
960 -> 1024 -> 512
```

This adapter would transform QVIM embeddings into the appropriate dimension for AudioLDM conditioning.

### Phase 3: Model Integration and Training

I assembled the modified model by:
- Replacing CLAP text conditioning with QVIM embedding-based conditioning
- Freezing all parameters except:
  - The adapter
  - Parameters named 'film' (conditioning mechanism)

For training, I implemented a variable guidance scale:
- Starting at 1.0
- Gradually increasing to 3.0 after several epochs

I used 50-100 DDIM steps with ddim_eta=1.0 for sampling during training.

## Challenges and Issues

### Issue 1: Minimal Training Progress

The initial training showed concerning patterns:
- Training loss did not meaningfully decrease
- Generated outputs were coherent sounds but unrelated to the conditioning

I noticed that increasing the learning rate had no significant effect on the training loss trajectory.

### Issue 2: Extremely Small Gradients

Upon investigation, I discovered extremely small gradients:
- Gradient magnitudes of 1e-7 to 1e-5 at the adapter and film layers
- This explained why learning rate adjustments weren't affecting training

### Issue 3: Catastrophic Forgetting

When I deactivated gradient clipping and incrementally increased the learning rate:
- The training loss initially maintained the same pattern
- Eventually, the loss spiked upward before beginning to decrease
- Generated audio samples became completely silent

This suggested catastrophic forgetting, likely related to the 'film' layers that handle conditioning.

### Issue 4: Adapter-Only Training Issues

I tried isolating the training to just the adapter:
- The loss trajectory remained largely unchanged even with learning rates increased to 0.1, 1, and 100
- At an extreme learning rate of 1e6, the trajectory was slightly higher but then suddenly produced NaN values

### Issue 5: CLAP Embedding Issues

I explored an alternative approach of training QVIM to mimic CLAP's embedding structure:
- Modified QVIM to produce 512-dimensional embeddings
- Used three cosine losses between {qvim_ref, qvim_imitation, clap_audio} with CLAP frozen
- Discovered that approximately 99% of CLAP embeddings were coming out as NaN values

A possible cause was identified: QVIM expects zero-padded audio to make samples 10 seconds long, while CLAP does not. Removing zero-padding before processing with CLAP resolved the NaN issue for approximately 50% of samples.

## Validation Tests

To isolate issues, I conducted several unit tests:
- Verified the VAE worked independently on the QVIM dataset
- Confirmed the model produced output when given a zero-imitation

This methodical testing helped identify component-specific issues vs. integration problems.

## Current Status

The alternative approach of training QVIM to align with CLAP embeddings appears most promising. Resolving the remaining NaN issues in CLAP embeddings is the current focus, as this approach avoids the instability observed when trying to adapt the diffusion model's conditioning mechanism directly.