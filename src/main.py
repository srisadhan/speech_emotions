import yaml
import sys
import wget 
import zipfile
import collections
import os 

from utils import skip_run
from pathlib import Path
from datasets.create_dataset import (download_dataset, extract_using_torchaudio,
                                    pad_zerosTo_waveforms, club_intensities_as_repetitions,
                                    extract_MelSpectrogram)

import torch 
import torchaudio

import matplotlib.pyplot as plt 
import deepdish as dd

# The configuration file
config = yaml.load(open('src/config.yaml'), Loader=yaml.SafeLoader)

# Create the required folders
filepath = str(Path(__file__).parents[1] / 'data')
if not os.path.isdir(filepath):
    os.mkdir(filepath)

filepath = str(Path(__file__).parents[1] / 'data/raw')
if not os.path.isdir(filepath):
    os.mkdir(filepath)

filepath = str(Path(__file__).parents[1] / 'data/interim')
if not os.path.isdir(filepath):
    os.mkdir(filepath)

filepath = str(Path(__file__).parents[1] / 'data/processed')
if not os.path.isdir(filepath):
    os.mkdir(filepath)

# ________________ RAVDESS dataset _______________ #
# download data from web and place it in the desired location
with skip_run('skip', 'download_RAVDESS_data') as check, check():
    url = config['url']
    download_dataset(url, config)

# import the data from .wav files and create a dictionary
with skip_run('run', 'import_audio_files') as check, check():
    extract_using_torchaudio(config)
    
# pad the audio files with zeros to make them of same size 
with skip_run('skip', 'pad_zeros_to_waveforms_for_constant_length') as check, check():
    pad_zerosTo_waveforms(config)


# WARNING: only run it if you want to neglect the change in intensities
# and copy in into repetitions
with skip_run('skip', 'include_intensities_as_repetitions') as check, check():
    club_intensities_as_repetitions(config)

with skip_run('run', 'import_audio_files') as check, check():
    extract_MelSpectrogram(config, True,False)
    