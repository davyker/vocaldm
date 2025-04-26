import glob
import os

import librosa
import numpy as np
import pandas as pd
import torch


class VimSketchDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            dataset_dir,
            sample_rate=32000,           # QVIM uses 32kHz sample rate 
            audioldm_sample_rate=16000,  # AudioLDM uses 16kHz sample rate
            duration=10.0,               # AudioLDM expects 10-second audio samples
            use_original_audioldm_mel=False  # Use AudioLDM's original mel processing pipeline
    ):
        self.dataset_dir = dataset_dir
        self.sample_rate = sample_rate
        self.audioldm_sample_rate = audioldm_sample_rate
        self.duration = duration
        self.use_original_audioldm_mel = use_original_audioldm_mel
        
        # Initialize AudioLDM's STFT processor if using original mel processing
        if self.use_original_audioldm_mel:
            from audioldm.audio import TacotronSTFT
            from audioldm.utils import default_audioldm_config
            
            # Get default configuration for AudioLDM
            config = default_audioldm_config()
            
            # Create STFT processor
            self.fn_STFT = TacotronSTFT(
                config["preprocessing"]["stft"]["filter_length"],
                config["preprocessing"]["stft"]["hop_length"],
                config["preprocessing"]["stft"]["win_length"],
                config["preprocessing"]["mel"]["n_mel_channels"],
                config["preprocessing"]["audio"]["sampling_rate"],
                config["preprocessing"]["mel"]["mel_fmin"],
                config["preprocessing"]["mel"]["mel_fmax"],
            )
            
            # Target length for AudioLDM's mel spectrograms (1024 frames for 10 seconds)
            self.target_length = config["preprocessing"]["mel"]["target_length"]

        reference_filenames = pd.read_csv(
            os.path.join(dataset_dir, 'reference_file_names.csv'),
            sep='\t',
            header=None,
            names=['filename']
        )
        reference_filenames['reference_id'] = reference_filenames['filename'].transform(
            lambda x: "_".join(x.split('_')[1:])
        )

        imitation_file_names = pd.read_csv(
            os.path.join(dataset_dir, 'vocal_imitation_file_names.csv'),
            sep='\t',
            header=None,
            names=['filename']
        )
        imitation_file_names['reference_id'] = imitation_file_names['filename'].transform(
            lambda x: "_".join(x.split('_')[1:])
        )

        self.all_pairs = imitation_file_names.merge(
            reference_filenames,
            left_on="reference_id",
            right_on="reference_id", how="left",
            suffixes=('_imitation', '_reference')
        )

        self.cached_files = {}
        self.cached_mels = {}  # Cache for mel spectrograms

    def load_audio(self, path):
        if path not in self.cached_files:
            audio, sr = librosa.load(
                path,
                sr=self.sample_rate,
                mono=True,
                duration=self.duration
            )
            self.cached_files[path] = audio
        return self.__pad_or_truncate__(self.cached_files[path])

    def get_mel_spectrogram(self, path):
        """Process audio file to mel spectrogram using AudioLDM's original pipeline"""
        if path not in self.cached_mels and self.use_original_audioldm_mel:
            from audioldm.audio.tools import wav_to_fbank
            
            # Use AudioLDM's original processing pipeline
            fbank, _, _ = wav_to_fbank(
                path, 
                target_length=self.target_length, 
                fn_STFT=self.fn_STFT
            )
            
            # Format as [channel, time, freq] as expected by AudioLDM
            # Do NOT add an extra batch dimension, as batching will be done by the DataLoader
            mel = fbank.unsqueeze(0)  # Add only channel dimension
            
            # Cache the processed mel
            self.cached_mels[path] = mel
            return mel
                
        return self.cached_mels.get(path, None)

    def __pad_or_truncate__(self, audio):
        fixed_length = int(self.sample_rate * self.duration)
        if len(audio) < fixed_length:
            array = np.zeros(fixed_length, dtype="float32")
            array[:len(audio)] = audio
        if len(audio) >= fixed_length:
            array = audio[:fixed_length]
        return array

    def __getitem__(self, index):
        row = self.all_pairs.iloc[index]
        
        # Get filenames
        reference_filename = row['filename_reference']
        imitation_filename = row['filename_imitation']
        
        # Extract class from filenames by removing everything up to first underscore
        # Example: "00003_000Animal_Domestic animals_pets_Cat_Growling" -> "000Animal_Domestic animals_pets_Cat_Growling"
        imitation_class = '_'.join(imitation_filename.split('_')[1:]) if '_' in imitation_filename else imitation_filename
        reference_class = '_'.join(reference_filename.split('_')[1:]) if '_' in reference_filename else reference_filename
        
        # Paths to audio files
        reference_path = os.path.join(self.dataset_dir, 'references', reference_filename)
        imitation_path = os.path.join(self.dataset_dir, 'vocal_imitations', imitation_filename)
        
        # Create return dictionary
        item = {
            'reference_filename': reference_filename,
            'imitation_filename': imitation_filename,
            'reference': self.load_audio(reference_path),
            'imitation': self.load_audio(imitation_path),
            'imitation_class': imitation_class,
            'reference_class': reference_class
        }
        
        # Add mel spectrograms if using AudioLDM's original processing
        if self.use_original_audioldm_mel:
            reference_mel = self.get_mel_spectrogram(reference_path)
            if reference_mel is not None:
                item['mel_reference'] = reference_mel
        
        return item

    def __len__(self):
        return len(self.all_pairs)


class AESAIMLA_DEV(torch.utils.data.Dataset):

    def __init__(
            self,
            dataset_dir,
            sample_rate=32000,  # QVIM uses 32kHz sample rate 
            duration=10.0       # AudioLDM expects 10-second audio samples
    ):
        self.dataset_dir = dataset_dir
        self.sample_rate = sample_rate
        self.duration = duration

        pairs = pd.read_csv(
            os.path.join(dataset_dir, 'DEV Dataset.csv'),
            skiprows=1
        )[['Label', 'Class', 'Items', 'Query 1', 'Query 2', 'Query 3']]

        # pairs.columns = pairs.columns.droplevel()

        pairs = pairs.melt(id_vars=[col for col in pairs.columns if "Query" not in col],
                           value_vars=["Query 1", "Query 2", "Query 3"],
                           var_name="Query Type",
                           value_name="Query")

        pairs = pairs.dropna()
        print("Total number of imitations: ", len(pairs["Query"].unique()))
        print("Total number of references: ", len(pairs["Items"].unique()))

        self.all_pairs = pairs
        self.check_files()

        print(f"Found {len(self.all_pairs)} pairs.")

        self.cached_files = {}


    def check_files(self):
        for i, pair in self.all_pairs.iterrows():
            reference_name = os.path.join(self.dataset_dir, 'Items', pair['Class'], pair['Items'])
            if not os.path.exists(reference_name):
                print("Missing: ", reference_name)
            imitation_name = os.path.join(self.dataset_dir, 'Queries', pair['Class'], pair['Query'])
            if not os.path.exists(imitation_name):
                print("Missing: ", imitation_name)

    def load_audio(self, path):
        if path not in self.cached_files:
            audio, sr = librosa.load(
                path,
                sr=self.sample_rate,
                mono=True,
                duration=self.duration
            )
            self.cached_files[path] = audio
        return self.__pad_or_truncate__(self.cached_files[path])



    def __pad_or_truncate__(self, audio):

        fixed_length = int(self.sample_rate * self.duration)
        array = np.zeros(fixed_length, dtype="float32")

        if len(audio) < fixed_length:
            array[:len(audio)] = audio
        if len(audio) >= fixed_length:
            array[:fixed_length]  = audio[:fixed_length]

        return array




    def __getitem__(self, index):

        row = self.all_pairs.iloc[index]

        reference_name = os.path.join(self.dataset_dir, 'Items', row['Class'], row['Items'])
        imitation_name = os.path.join(self.dataset_dir, 'Queries', row['Class'], row['Query'])

        return {
            'reference_filename': row['Items'],
            'imitation_filename': row['Query'],
            'reference': self.load_audio(reference_name),
            'imitation': self.load_audio(imitation_name),
            'reference_class': row['Class'],
            'imitation_class': row['Class']
        }

    def __len__(self):
        return len(self.all_pairs)
