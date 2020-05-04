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
from models.emotionet_conv import EMOTIONET

import torch 
import torchaudio

import matplotlib.pyplot as plt 
import deepdish as dd
import argparse
from torch.utils.tensorboard import SummaryWriter

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
with skip_run('skip', 'import_audio_files') as check, check():
    extract_using_torchaudio(config)
    
# pad the audio files with zeros to make them of same size 
with skip_run('skip', 'pad_zeros_to_waveforms_for_constant_length') as check, check():
    pad_zerosTo_waveforms(config)


# WARNING: only run it if you want to neglect the change in intensities
# and copy in into repetitions
with skip_run('skip', 'include_intensities_as_repetitions') as check, check():
    club_intensities_as_repetitions(config)

with skip_run('skip', 'extract_MelSpec_and_save') as check, check():
    extract_MelSpectrogram(config, True,False)

with skip_run('run', 'train_torch_model') as check, check():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_ = torch.nn.CrossEntropyLoss(reduction='sum')
    encoder = EMOTIONET(dataset_train='speech1_MelSpec',
                        dataset_val='speech1_MelSpec',
                        device=device,
                        loss_=loss_,
                        load_model=False,
                        beta=1.0).to(device=device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    encoder.opt = optimizer
    # cpt = args.cpt
    # if args.load_model:
    #     encoder.load_model_cpt(cpt=cpt, device=device)

    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lr_lambda=lambda x: x*0.95)

    lr_scheduler = None
    encoder.train_loop(optimizer,
                        lr_scheduler,
                        loss_,
                        batch_size=encoder.config['batch_size'],
                        cpt=None)