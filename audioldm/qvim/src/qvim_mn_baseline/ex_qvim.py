import argparse
import os
import math
import copy
import platform

from copy import deepcopy
import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
import pytorch_lightning as pl

# Enable Tensor Cores for faster training with minimal precision loss
torch.set_float32_matmul_precision('high')

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor

from audioldm.qvim.src.qvim_mn_baseline.dataset import VimSketchDataset, AESAIMLA_DEV
from audioldm.qvim.src.qvim_mn_baseline.download import download_vimsketch_dataset, download_qvim_dev_dataset
from audioldm.qvim.src.qvim_mn_baseline.mn.preprocess import AugmentMelSTFT
from audioldm.qvim.src.qvim_mn_baseline.mn.model import get_model as get_mobilenet
from audioldm.qvim.src.qvim_mn_baseline.utils import NAME_TO_WIDTH
from audioldm.qvim.src.qvim_mn_baseline.metrics import compute_mrr, compute_ndcg

class QVIMModule(pl.LightningModule):
    """
    Pytorch Lightning Module for the QVIM Model
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.mel = AugmentMelSTFT(
            n_mels=config.n_mels,
            sr=config.sample_rate,
            win_length=config.window_size,
            hopsize=config.hop_size,
            n_fft=config.n_fft,
            freqm=config.freqm,
            timem=config.timem,
            fmin=config.fmin,
            fmax=config.fmax,
            fmin_aug_range=config.fmin_aug_range,
            fmax_aug_range=config.fmax_aug_range
        )

        # get the to be specified mobilenetV3 as encoder
        self.imitation_encoder = get_mobilenet(
            width_mult=NAME_TO_WIDTH(config.pretrained_name),
            pretrained_name=config.pretrained_name
        )

        self.reference_encoder = deepcopy(self.imitation_encoder)

        initial_tau = torch.zeros((1,)) + config.initial_tau
        self.tau = torch.nn.Parameter(initial_tau, requires_grad=config.tau_trainable)

        self.validation_output = []

    def forward(self, queries, items):
        return self.forward_imitation(queries), self.forward_reference(items)

    def forward_imitation(self, imitations, enable_timing=False):
        # Process MEL spectrogram generation with timing if enabled
        
        if enable_timing:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            
        with torch.no_grad():
            imitations = self.mel(imitations).unsqueeze(1)
            
        if enable_timing:
            end.record()
            torch.cuda.synchronize()
            print(f"MEL processing time: {start.elapsed_time(end):.2f} ms")
            start.record()
            
        y_imitation = self.imitation_encoder(imitations)[1]
        
        if enable_timing:
            end.record()
            torch.cuda.synchronize()
            print(f"Encoder time: {start.elapsed_time(end):.2f} ms")
            start.record()
            
        y_imitation = torch.nn.functional.normalize(y_imitation, dim=1)
        
        if enable_timing:
            end.record()
            torch.cuda.synchronize()
            print(f"Normalization time: {start.elapsed_time(end):.2f} ms")
            
        return y_imitation

    def forward_reference(self, items, enable_timing=False):
        # Process MEL spectrogram generation with timing if enabled
        
        if enable_timing:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            
        with torch.no_grad():
            items = self.mel(items).unsqueeze(1)
            
        if enable_timing:
            end.record()
            torch.cuda.synchronize()
            print(f"MEL processing time: {start.elapsed_time(end):.2f} ms")
            start.record()
            
        y_reference = self.reference_encoder(items)[1]
        
        if enable_timing:
            end.record()
            torch.cuda.synchronize()
            print(f"Encoder time: {start.elapsed_time(end):.2f} ms")
            start.record()
            
        y_reference = torch.nn.functional.normalize(y_reference, dim=1)
        
        if enable_timing:
            end.record()
            torch.cuda.synchronize()
            print(f"Normalization time: {start.elapsed_time(end):.2f} ms")
            
        return y_reference

    def training_step(self, batch, batch_idx):
        # Store current batch for use in lr_scheduler_step
        self.current_batch = batch
        
        self.lr_scheduler_step(batch_idx)

        y_imitation, y_reference = self.forward(batch['imitation'], batch['reference'])

        C = torch.matmul(y_imitation, y_reference.T)
        C = C / torch.abs(self.tau)

        C_text = torch.log_softmax(C, dim=1)

        paths = np.array([hash(p) for i, p in enumerate(batch['imitation_filename'])])
        I = torch.tensor(paths[None, :] == paths[:, None])


        loss = - C_text[torch.where(I)].mean()

        # Show train loss in progress bar without second pass
        self.log('train/loss', loss, prog_bar=True, batch_size=len(batch['imitation']))
        self.log('train/tau', self.tau, batch_size=len(batch['imitation']))
        self.log('lr', self.optimizers().param_groups[0]['lr'], prog_bar=True, batch_size=len(batch['imitation']))

        return loss

    def validation_step(self, batch, batch_idx):

        y_imitation, y_reference = self.forward(batch['imitation'], batch['reference'])

        C = torch.matmul(y_imitation, y_reference.T)
        C = C / torch.abs(self.tau)

        C_text = torch.log_softmax(C, dim=1)

        paths = np.array([hash(p) for i, p in enumerate(batch['imitation_filename'])])
        I = torch.tensor(paths[None, :] == paths[:, None])


        loss = - C_text[torch.where(I)].mean()

        self.log('val/loss', loss, prog_bar=True, batch_size=len(batch['imitation']))
        self.log('val/tau', self.tau, batch_size=len(batch['imitation']))


        self.validation_output.extend([
            {
                'imitation': copy.deepcopy(y_imitation.detach().cpu().numpy()),
                'reference': copy.deepcopy(y_reference.detach().cpu().numpy()),
                'imitation_filename': batch['imitation_filename'],
                'reference_filename': batch['reference_filename'],
                'imitation_class': batch['imitation_class'],
                'reference_class': batch['reference_class']
            }
        ])

    def on_validation_epoch_end(self):
        validation_output = self.validation_output

        # Concatenate imitation and reference arrays
        imitations = np.concatenate([b['imitation'] for b in validation_output])
        reference = np.concatenate([b['reference'] for b in validation_output])

        # Flatten filenames lists
        imitation_filenames = sum([b['imitation_filename'] for b in validation_output], [])
        reference_filenames = sum([b['reference_filename'] for b in validation_output], [])

        # Compute new ground truth based on classes
        imitation_classes = sum([b['imitation_class'] for b in validation_output], [])
        reference_classes = sum([b['reference_class'] for b in validation_output], [])

        # Generate ground truth mapping
        ground_truth_mrr = {fi: rf for fi, rf in zip(imitation_filenames, reference_filenames)}

        # Compute similarity scores using matrix multiplication
        # Remove duplicates in reference vectors and filenames
        _, unique_indices = np.unique(reference_filenames, return_index=True)
        reference = reference[unique_indices]
        reference_filenames = [reference_filenames[i] for i in unique_indices.tolist()]
        reference_classes = [reference_classes[i] for i in unique_indices.tolist()]

        ground_truth_classes = {
            ifn: [rfn for rfn, rfc in zip(reference_filenames, reference_classes) if rfc == ifc]
            for ifn, ifc in zip(imitation_filenames, imitation_classes)
        }

        scores_matrix = np.dot(imitations, reference.T)
        similarity_df = pd.DataFrame(scores_matrix, index=imitation_filenames, columns=reference_filenames)



        mrr = compute_mrr(similarity_df, ground_truth_mrr)
        ndcg = compute_ndcg(similarity_df, ground_truth_classes)

        # These are calculated across the entire validation set, so we use the total size
        total_validation_samples = len(imitation_filenames)
        self.log('val/mrr', mrr, prog_bar=True, batch_size=total_validation_samples)
        self.log('val/ndcg', ndcg, prog_bar=True, batch_size=total_validation_samples)

        # clear the cached outputs
        self.validation_output = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.config.weight_decay,
            amsgrad=False
        )
        
        if self.config.lr_schedule == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",  # We want to maximize MRR
                factor=0.5,  # Reduce LR by half when plateauing
                patience=2,  # Number of epochs with no improvement
                verbose=True,
                min_lr=self.config.min_lr
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/mrr",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        elif self.config.lr_schedule == "cosine_annealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.n_epochs,
                eta_min=self.config.min_lr,
                verbose=True
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        elif self.config.lr_schedule == "cosine":
            # Use original manual scheduling
            return optimizer
        else:
            # Default to original scheduling
            return optimizer

    def lr_scheduler_step(self, batch_idx, optimizer_idx=None, scheduler_idx=None):
        # Skip if using plateau scheduler
        if self.config.lr_schedule == "plateau":
            return
            
        steps_per_epoch = self.trainer.num_training_batches

        min_lr = self.config.min_lr
        max_lr = self.config.max_lr
        current_step = self.current_epoch * steps_per_epoch + batch_idx
        warmup_steps = self.config.warmup_epochs * steps_per_epoch
        total_steps = (self.config.warmup_epochs + self.config.rampdown_epochs) * steps_per_epoch
        decay_steps = total_steps - warmup_steps

        if current_step < warmup_steps:
            # Linear warmup
            lr = min_lr + (max_lr - min_lr) * (current_step / warmup_steps)
        elif current_step < total_steps:
            # Cosine decay
            decay_progress = (current_step - warmup_steps) / decay_steps
            lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * decay_progress))
        else:
            # Constant learning rate
            lr = min_lr

        for param_group in self.optimizers(use_pl_optimizer=False).param_groups:
            param_group['lr'] = lr

        self.log('train/lr', lr, sync_dist=True)


def train(config):
    # Train dual encoder for QBV

    # download the data set if the folder does not exist
    download_vimsketch_dataset(config.dataset_path)
    download_qvim_dev_dataset(config.dataset_path)

    wandb_logger = WandbLogger(
        project=config.project,
        config=config
    )

    # Load the VimSketch dataset
    full_ds = VimSketchDataset(
        os.path.join(config.dataset_path, 'Vim_Sketch_Dataset'),
        sample_rate=config.sample_rate,
        duration=config.duration
    )
    
    # Create train/validation split from same dataset (better for diffusion conditioning)
    dataset_size = len(full_ds)
    val_size = int(dataset_size * 0.15)  # Use 15% of data for validation
    train_size = dataset_size - val_size
    
    # Use random_split for better generalization
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducible split
    )
    
    print(f"Training with {train_size} samples, validating with {val_size} samples from same distribution")
    
    # Keep the original evaluation dataset for final evaluation
    final_eval_ds = AESAIMLA_DEV(
        os.path.join(config.dataset_path, 'qvim-dev'),
        sample_rate=config.sample_rate,
        duration=config.duration
    )

    train_dl = DataLoader(
        dataset=train_ds,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=False,          # Reduced memory usage
        persistent_workers=False,  # Allow worker cleanup between batches
        prefetch_factor=1 if config.num_workers > 0 else None          # Prefetch 1 batch per worker (reduced for memory)
    )

    val_dl = DataLoader(
        dataset=val_ds,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=1 if config.num_workers > 0 else None
    )
    
    final_eval_dl = DataLoader(
        dataset=final_eval_ds,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=1 if config.num_workers > 0 else None
    )

    pl_module = QVIMModule(config)

    callbacks = [LearningRateMonitor(logging_interval='step')]
    
    # Add early stopping based on validation loss (better for diffusion conditioning)
    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,  # Can be set via command line
        verbose=True
    )
    callbacks.append(early_stop_callback)
    
    if config.model_save_path:
        # Primary checkpoint: Best validation loss (for diffusion conditioning quality)
        callbacks.append(
            ModelCheckpoint(
            dirpath=os.path.join(config.model_save_path, wandb_logger.experiment.name),
            filename="best-loss-checkpoint",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            verbose=True
            )
        )
        
        # Secondary checkpoint: Best MRR (for reference)
        callbacks.append(
            ModelCheckpoint(
            dirpath=os.path.join(config.model_save_path, wandb_logger.experiment.name),
            filename="best-mrr-checkpoint",
            monitor="val/mrr",
            mode="max",
            save_top_k=1,
            verbose=True
            )
        )
        
        # Save last checkpoint
        callbacks.append(
            ModelCheckpoint(
            dirpath=os.path.join(config.model_save_path, wandb_logger.experiment.name),
            filename="last-checkpoint",
            save_last=True,
            verbose=False
            )
        )
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        logger=wandb_logger,
        accelerator='auto',
        precision=16,      # Enable mixed precision for faster training
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=1,
        enable_checkpointing=True
    )

    # Determine if we're continuing training from a checkpoint
    ckpt_path = None
    if config.continue_from:
        ckpt_path = config.continue_from
        print(f"\n----- Continuing training from checkpoint: {ckpt_path} -----")
        # Load only the model weights from checkpoint, not callback states
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        pl_module.load_state_dict(checkpoint["state_dict"])
        print("Model weights loaded from checkpoint (without callback states)")
    
    # Initial validation on in-domain validation set
    trainer.validate(
        pl_module,
        dataloaders=val_dl
    )

    # Train with in-domain validation
    # No need to pass ckpt_path if we've already loaded the model above
    trainer.fit(
        pl_module,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl
    )
    
    # Final evaluation on selected dataset
    if config.final_eval_dataset == "dev":
        print("\n----- Final Evaluation on QVIM-DEV Dataset -----")
        eval_dataloader = final_eval_dl
    else:
        print("\n----- Final Evaluation on VimSketch Validation Split -----")
        eval_dataloader = val_dl
        
    trainer.validate(
        pl_module,
        dataloaders=eval_dataloader
    )

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Argument parser for training the QVIM model.")

    # General
    parser.add_argument('--project', type=str, default="qvim",
                        help="Project name in wandb.")
    parser.add_argument('--num_workers', type=int, default=16,
                        help="Number of data loader workers. Reduced for WSL memory stability. Set to 0 for no multiprocessing.")
    parser.add_argument('--num_gpus', type=int, default=1,
                        help="Number of GPUs to use for training.")
    parser.add_argument('--model_save_path', type=str, default=None,
                        help="Path to store the checkpoints. Use None to disable saving.")
    parser.add_argument('--dataset_path', type=str, default='audioldm/qvim/data',
                        help="Path to the data sets.")

    # Encoder architecture
    parser.add_argument('--pretrained_name', type=str, default="mn10_as",
                        help="Pretrained model name for transfer learning.")

    # Training
    parser.add_argument('--random_seed', type=int, default=None,
                        help="A seed to make the experiment reproducible. Set to None to disable.")
    parser.add_argument('--continue_from', type=str, default=None,
                        help="Path to checkpoint file to continue training from")
    parser.add_argument('--final_eval_dataset', type=str, default="val", choices=["dev", "val"],
                        help="Dataset to use for final evaluation: 'dev' (QVIM-DEV) or 'val' (VimSketch val split)")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="Number of samples per batch.")
    parser.add_argument('--n_epochs', type=int, default=100,
                        help="Maximum number of training epochs (can stop earlier with early stopping).")
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0,
                        help="Minimum change in the monitored metric to qualify as an improvement.")
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help="L2 weight regularization to prevent overfitting.")
    parser.add_argument('--max_lr', type=float, default=0.0003,
                        help="Maximum learning rate.")
    parser.add_argument('--min_lr', type=float, default=0.0001,
                        help="Final learning rate at the end of training.")
    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help="Number of warm-up epochs where learning rate increases gradually.")
    parser.add_argument('--rampdown_epochs', type=int, default=17,
                        help="Duration (in epochs) for learning rate ramp-down.")
    parser.add_argument('--initial_tau', type=float, default=0.07,
                        help="Temperature parameter for the loss function.")
    parser.add_argument('--tau_trainable', default=True, action='store_true',
                        help="make tau trainable or not.")
    parser.add_argument('--lr_schedule', type=str, default="cosine", choices=["cosine", "plateau", "cosine_annealing"],
                        help="Learning rate schedule: 'cosine' (original), 'plateau' (reduce on plateau), or 'cosine_annealing' (smoother decay)")

    # Preprocessing
    parser.add_argument('--duration', type=float, default=10.0,
                        help="Duration of audio clips in seconds.")

    # Spectrogram Parameters
    parser.add_argument('--sample_rate', type=int, default=32000,
                        help="Target sampling rate for audio resampling.")
    parser.add_argument('--window_size', type=int, default=800,
                        help="Size of the window for STFT in samples.")
    parser.add_argument('--hop_size', type=int, default=320,
                        help="Hop length for STFT in samples.")
    parser.add_argument('--n_fft', type=int, default=1024,
                        help="Number of FFT bins for spectral analysis.")
    parser.add_argument('--n_mels', type=int, default=128,
                        help="Number of mel filter banks for Mel spectrogram conversion.")
    parser.add_argument('--freqm', type=int, default=8,
                        help="Frequency masking parameter for spectrogram augmentation.")
    parser.add_argument('--timem', type=int, default=300,
                        help="Time masking parameter for spectrogram augmentation.")
    parser.add_argument('--fmin', type=int, default=0,
                        help="Minimum frequency cutoff for Mel spectrogram.")
    parser.add_argument('--fmax', type=int, default=None,
                        help="Maximum frequency cutoff for Mel spectrogram (None means use Nyquist frequency).")
    parser.add_argument('--fmin_aug_range', type=int, default=10,
                        help="Variation range for fmin augmentation.")
    parser.add_argument('--fmax_aug_range', type=int, default=2000,
                        help="Variation range for fmax augmentation.")

    args = parser.parse_args()

    if args.random_seed:
        pl.seed_everything(args.random_seed)

    train(args)
