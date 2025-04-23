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
            sample_rate=32000,  # QVIM uses 32kHz sample rate 
            duration=10.0       # AudioLDM expects 10-second audio samples
    ):
        self.dataset_dir = dataset_dir
        self.sample_rate = sample_rate
        self.duration = duration


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

        return {
            'reference_filename': reference_filename,
            'imitation_filename': imitation_filename,
            'reference': self.load_audio(os.path.join(self.dataset_dir, 'references', reference_filename)),
            'imitation': self.load_audio(os.path.join(self.dataset_dir, 'vocal_imitations', imitation_filename)),
            'imitation_class': imitation_class,
            'reference_class': reference_class
        }

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
