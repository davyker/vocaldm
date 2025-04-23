#!/usr/bin/env python
import argparse
import os
import random
import shutil
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from audioldm.qvim.src.qvim_mn_baseline.dataset import VimSketchDataset, AESAIMLA_DEV
from audioldm.qvim.src.qvim_mn_baseline.ex_qvim import QVIMModule

def find_checkpoint(run_folder):
    """Find best checkpoint in run folder"""
    for pattern in ["best-loss-checkpoint.ckpt", "best-mrr-checkpoint.ckpt", 
                   "best_checkpoint.ckpt", "best-checkpoint.ckpt", "*.ckpt"]:
        checkpoints = list(Path(run_folder).glob(f"**/{pattern}"))
        if checkpoints:
            return str(checkpoints[0])
    raise FileNotFoundError(f"No checkpoint found in {run_folder}")

def get_query_embedding(model, query_audio, device):
    """Get embedding for a query"""
    audio_tensor = torch.tensor(query_audio).float().unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.forward_imitation(audio_tensor)
    return embedding.cpu().numpy()[0]

def get_reference_embeddings(model, dataset, device):
    """Get embeddings for unique reference files"""
    # Set batch size for better GPU utilization
    batch_size = 32  # Adjust based on your GPU
    
    # Get unique references and build index mapping
    if hasattr(dataset, 'all_pairs') and 'filename_reference' in dataset.all_pairs.columns:
        # VimSketch dataset - use metadata
        unique_refs = dataset.all_pairs['filename_reference'].unique().tolist()
        print(f"Processing {len(unique_refs)} unique references...")
        
        # Build file to index mapping
        lookup_cache = {}
        for j in tqdm(range(len(dataset)), desc="Building index"):
            ref_file = dataset[j]['reference_filename']
            if ref_file in unique_refs and ref_file not in lookup_cache:
                lookup_cache[ref_file] = j
        
        # Keep only references we found in the dataset
        valid_refs = [ref for ref in unique_refs if ref in lookup_cache]
        ref_to_idx = lookup_cache
        
    else:
        # Fallback for non-VimSketch datasets
        ref_to_idx = {}
        for i in tqdm(range(len(dataset)), desc="Indexing dataset"):
            ref_file = dataset[i]['reference_filename']
            if ref_file not in ref_to_idx:
                ref_to_idx[ref_file] = i
                
        valid_refs = list(ref_to_idx.keys())
        print(f"Found {len(valid_refs)} unique references")
    
    # Process in batches (same for both dataset types)
    ref_embeddings = {}
    batches = [valid_refs[i:i+batch_size] for i in range(0, len(valid_refs), batch_size)]
    
    # Process each batch
    for batch_refs in tqdm(batches, desc="Processing batches"):
        batch_audio = []
        
        # Collect audio for batch
        for ref_file in batch_refs:
            idx = ref_to_idx[ref_file]
            batch_audio.append(dataset[idx]['reference'])
        
        # Process the batch
        batch_tensor = torch.tensor(np.array(batch_audio)).float().to(device)
        with torch.no_grad():
            batch_embeddings = model.forward_reference(batch_tensor)
        
        # Store results
        for i, ref_file in enumerate(batch_refs):
            ref_embeddings[ref_file] = batch_embeddings[i].cpu().numpy()
    
    return ref_embeddings
    return ref_embeddings

def save_results(query_file, true_match, top_matches, dataset, output_dir):
    """Save query, true match, and top matches to output directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy query file
    if dataset.dataset_dir.endswith('Vim_Sketch_Dataset'):
        query_path = os.path.join(dataset.dataset_dir, 'vocal_imitations', query_file)
        true_match_path = os.path.join(dataset.dataset_dir, 'references', true_match)
    else:  # AESAIMLA_DEV
        query_class = query_file.split('_')[0] 
        query_path = os.path.join(dataset.dataset_dir, 'Queries', query_class, query_file)
        true_match_class = true_match.split('_')[0]
        true_match_path = os.path.join(dataset.dataset_dir, 'Items', true_match_class, true_match)
    
    # Save query and true match
    shutil.copy(query_path, os.path.join(output_dir, f"imitation_{query_file}"))
    shutil.copy(true_match_path, os.path.join(output_dir, f"true_match_{true_match}"))
    
    # Create matches directory
    matches_dir = os.path.join(output_dir, "top_matches")
    os.makedirs(matches_dir, exist_ok=True)
    
    # Copy top matches
    for i, (ref_file, similarity) in enumerate(top_matches):
        if dataset.dataset_dir.endswith('Vim_Sketch_Dataset'):
            source_path = os.path.join(dataset.dataset_dir, 'references', ref_file)
        else:  # AESAIMLA_DEV
            ref_class = ref_file.split('_')[0]
            source_path = os.path.join(dataset.dataset_dir, 'Items', ref_class, ref_file)
        
        # Create destination filename with rank and similarity
        dest_filename = f"{i+1:02d}_{similarity:.4f}_{ref_file}"
        dest_path = os.path.join(matches_dir, dest_filename)
        
        # Copy file
        shutil.copy(source_path, dest_path)
    
    # Create summary file
    true_match_rank = None
    for i, (ref_file, _) in enumerate(top_matches):
        if ref_file == true_match:
            true_match_rank = i + 1
            break
    
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(f"Imitation: {query_file}\n")
        f.write(f"True reference: {true_match}\n")
        if true_match_rank:
            f.write(f"True match ranked: {true_match_rank} out of {len(top_matches)}\n")
        else:
            f.write(f"True match not in top {len(top_matches)}\n")

def main():
    parser = argparse.ArgumentParser(description="Simple query embedding retrieval")
    parser.add_argument('--run_folder', type=str, required=True,
                       help="Path to trained model run folder")
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument('--query_file', type=str,
                           help="Specific imitation file to query")
    query_group.add_argument('--random_query', type=str, choices=['vimsketch', 'aesaimla'],
                           help="Select random query from dataset")
    parser.add_argument('--dataset_path', type=str, default='data',
                       help="Path to datasets")
    parser.add_argument('--output_dir', type=str, default='single_query_tests',
                       help="Base directory for output")
    parser.add_argument('--top_n', type=int, default=10,
                       help="Number of top matches to retrieve")
    args = parser.parse_args()
    
    # Create output directory
    run_name = os.path.basename(os.path.normpath(args.run_folder))
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, run_name, timestamp)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    if args.random_query == 'vimsketch' or (args.query_file and not args.query_file.startswith("Query")):
        dataset = VimSketchDataset(
            os.path.join(args.dataset_path, 'Vim_Sketch_Dataset'),
            sample_rate=32000,
            duration=10.0
        )
    else:
        dataset = AESAIMLA_DEV(
            os.path.join(args.dataset_path, 'qvim-dev'),
            sample_rate=32000,
            duration=10.0
        )
    
    # Get query
    if args.query_file:
        query_file = args.query_file
        # Find sample with this filename
        for i in range(len(dataset)):
            if dataset[i]['imitation_filename'] == query_file:
                query_sample = dataset[i]
                break
        else:
            raise ValueError(f"Query file {query_file} not found in dataset")
    else:
        # Random query
        idx = random.randint(0, len(dataset) - 1)
        query_sample = dataset[idx]
        query_file = query_sample['imitation_filename']
    
    print(f"Query: {query_file}")
    
    # Find true match
    if hasattr(dataset, 'all_pairs'):
        # VimSketch dataset
        match_row = dataset.all_pairs[dataset.all_pairs['filename_imitation'] == query_file]
        true_match = match_row['filename_reference'].iloc[0]
    else:
        # AESAIMLA dataset
        true_match = query_sample['reference_filename']
    
    print(f"True match: {true_match}")
    
    # Load model
    print(f"Loading model from {args.run_folder}")
    ckpt_path = find_checkpoint(args.run_folder)
    print(f"Using checkpoint: {ckpt_path}")
    
    # Create minimal config
    class Config:
        def __init__(self):
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
            self.initial_tau = 0.07
            self.tau_trainable = False
    
    # Load model
    model = QVIMModule.load_from_checkpoint(ckpt_path, config=Config())
    model.to(device)
    model.eval()
    
    # Get query embedding
    query_embedding = get_query_embedding(model, query_sample['imitation'], device)
    
    # Get reference embeddings 
    ref_embeddings = get_reference_embeddings(model, dataset, device)
    
    # Compute similarities
    similarities = {}
    for ref_file, embedding in ref_embeddings.items():
        sim = np.dot(query_embedding, embedding)
        similarities[ref_file] = sim
    
    # Get all matches sorted by similarity
    all_sorted_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    # Find rank of true match
    true_match_rank = None
    for i, (ref_file, _) in enumerate(all_sorted_matches):
        if ref_file == true_match:
            true_match_rank = i + 1
            break
    
    # Get top N matches for display
    top_matches = all_sorted_matches[:args.top_n]
    
    # Print rank information
    if true_match_rank:
        print(f"\nTrue match '{true_match}' ranked: {true_match_rank} out of {len(all_sorted_matches)} references")
        if true_match_rank <= args.top_n:
            print(f"True match is in the top {args.top_n} (position {true_match_rank})")
        else:
            print(f"True match is NOT in the top {args.top_n} (position {true_match_rank})")
    else:
        print(f"\nWarning: True match '{true_match}' not found in results")
    
    # Print top matches
    print(f"\nTop {args.top_n} matches:")
    for i, (ref_file, similarity) in enumerate(top_matches):
        is_true_match = "(TRUE MATCH)" if ref_file == true_match else ""
        print(f"{i+1}. {ref_file} (similarity: {similarity:.4f}) {is_true_match}")
    
    # Save results
    save_results(query_file, true_match, top_matches, dataset, output_dir)
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()