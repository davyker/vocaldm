#!/usr/bin/env python3
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import glob

def main():
    # Define paths to audio files
    dataset_path = "audioldm/qvim/data/Vim_Sketch_Dataset"
    imitation_path = os.path.join(dataset_path, "vocal_imitations")
    reference_path = os.path.join(dataset_path, "references")
    
    print(f"Analyzing audio files from {dataset_path}")
    
    # Get all audio files
    imitation_files = glob.glob(os.path.join(imitation_path, "*.wav"))
    reference_files = glob.glob(os.path.join(reference_path, "*.wav"))
    
    if not imitation_files or not reference_files:
        print(f"Error: No audio files found in {imitation_path} or {reference_path}")
        print(f"Imitation files found: {len(imitation_files)}")
        print(f"Reference files found: {len(reference_files)}")
        
        # Check alternate paths
        alternate_paths = glob.glob(os.path.join(dataset_path, "**/"))
        print(f"Available directories in dataset: {alternate_paths}")
        return
    
    print(f"Found {len(imitation_files)} imitation files and {len(reference_files)} reference files")
    
    # Analyze file lengths
    imitation_lengths = []
    reference_lengths = []
    sample_rate = 32000  # Expected sample rate
    
    print("Analyzing imitation files...")
    for file_path in tqdm(imitation_files):
        try:
            # Use librosa to get duration directly
            duration = librosa.get_duration(filename=file_path)
            imitation_lengths.append(duration)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print("Analyzing reference files...")
    for file_path in tqdm(reference_files):
        try:
            # Use librosa to get duration directly
            duration = librosa.get_duration(filename=file_path)
            reference_lengths.append(duration)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Convert to numpy arrays
    imitation_lengths = np.array(imitation_lengths)
    reference_lengths = np.array(reference_lengths)
    
    # Print statistics
    print("\nImitation Lengths (seconds):")
    print(f"  Min: {np.min(imitation_lengths):.2f}")
    print(f"  Max: {np.max(imitation_lengths):.2f}")
    print(f"  Mean: {np.mean(imitation_lengths):.2f}")
    print(f"  Median: {np.median(imitation_lengths):.2f}")
    print(f"  Std Dev: {np.std(imitation_lengths):.2f}")
    
    print("\nReference Lengths (seconds):")
    print(f"  Min: {np.min(reference_lengths):.2f}")
    print(f"  Max: {np.max(reference_lengths):.2f}")
    print(f"  Mean: {np.mean(reference_lengths):.2f}")
    print(f"  Median: {np.median(reference_lengths):.2f}")
    print(f"  Std Dev: {np.std(reference_lengths):.2f}")
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # Imitation lengths histogram
    plt.subplot(2, 2, 1)
    plt.hist(imitation_lengths, bins=30, alpha=0.7, color='blue')
    plt.axvline(x=5.0, color='red', linestyle='--', label='Current duration (5s)')
    plt.title('Vocal Imitation Lengths')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.legend()
    
    # Reference lengths histogram
    plt.subplot(2, 2, 2)
    plt.hist(reference_lengths, bins=30, alpha=0.7, color='green')
    plt.axvline(x=5.0, color='red', linestyle='--', label='Current duration (5s)')
    plt.title('Reference Sound Lengths')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.legend()
    
    # Joint 2D histogram - only if we can match imitations to references
    if len(imitation_lengths) == len(reference_lengths):
        plt.subplot(2, 1, 2)
        plt.hist2d(imitation_lengths, reference_lengths, bins=30, cmap='viridis')
        plt.colorbar(label='Count')
        plt.axvline(x=5.0, color='red', linestyle='--', label='Current imitation cutoff')
        plt.axhline(y=5.0, color='red', linestyle='--', label='Current reference cutoff')
        plt.title('Joint Distribution of Audio Lengths')
        plt.xlabel('Imitation Length (seconds)')
        plt.ylabel('Reference Length (seconds)')
        plt.legend()
    else:
        plt.subplot(2, 1, 2)
        plt.text(0.5, 0.5, "Cannot create joint histogram - mismatched file counts", 
                horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('audio_length_analysis.png', dpi=300)
    print("\nPlots saved to audio_length_analysis.png")
    
    # Also save a summary CSV
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    summary = {
        'statistic': ['min', 'max', 'mean', 'median', 'std'] + [f'{p}th percentile' for p in percentiles],
        'imitation': [
            np.min(imitation_lengths),
            np.max(imitation_lengths),
            np.mean(imitation_lengths),
            np.median(imitation_lengths),
            np.std(imitation_lengths)
        ] + [np.percentile(imitation_lengths, p) for p in percentiles],
        'reference': [
            np.min(reference_lengths),
            np.max(reference_lengths),
            np.mean(reference_lengths),
            np.median(reference_lengths),
            np.std(reference_lengths)
        ] + [np.percentile(reference_lengths, p) for p in percentiles]
    }
    
    df = pd.DataFrame(summary)
    df.to_csv('audio_length_summary.csv', index=False)
    print("Summary statistics saved to audio_length_summary.csv")

if __name__ == "__main__":
    main()