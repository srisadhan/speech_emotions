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
                                    extract_MelSpectrogram, constant_shaped_spectrogram)
from models.emotionet_vae import EmotionNet_VAE, EmotionNet_VAE2
from models.speaker_encoder import SPEAKER_ENCODER

import torch 
import torchaudio

import matplotlib.pyplot as plt 
import deepdish as dd
import argparse
from torch.utils.tensorboard import SummaryWriter
import librosa
import soundfile
import numpy as np
import umap 

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

######################################################
## RAVDESS dataset
######################################################
# download data from web and place it in the desired location
with skip_run('skip', 'download_RAVDESS_data') as check, check():
    url = config['url']
    download_dataset(url, config)

run_str = 'skip'
with skip_run(run_str, 'Data_processing') as check, check():
    # import the data from .wav files and create a dictionary
    with skip_run(run_str, 'import_audio_files') as check, check():
        extract_using_torchaudio(config)
    
    with skip_run(run_str, 'extract_MelSpec_and_save') as check, check():
        extract_MelSpectrogram(config, True, False)
        # extract_MelSpectrogram(config, True, True)
    
    with skip_run(run_str, 'segment_speech_to_constant_length_MelSpec') as check, check():
        # load the previously extracted Mel Spectrograms
        data = dd.io.load((Path(__file__).parents[1] / config['speech1_MelSpec']))
    
        features, labels, speaker_id = constant_shaped_spectrogram(data, config)
        
        Data = collections.defaultdict()
        Data['features'] = features
        Data['labels']   = labels
        Data['speaker_id'] = speaker_id
        dd.io.save(str(Path(__file__).parents[1] / config['const_MelSpec1']), Data)
    
     #--------------------------------------------------------------------------------------#
    # NOTE: Deprecated, the following functionalities are implemented in different functions
    # only run it if you want to neglect the change in intensities and copy in into repetitions 
    # with skip_run('skip', 'pad_zeros_to_waveforms_for_constant_length_and_include_intensities_as_repetitions') as check, check():
    #     pad_zerosTo_waveforms(config)
    #     club_intensities_as_repetitions(config)
    #--------------------------------------------------------------------------------------#          
               

######################################################
## Training an Variation Auto Encoder (VAE)
######################################################         
with skip_run('skip', 'train_torch_model') as check, check():    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Training the model on: {}".format(device))
    
    # NOTE: EmotionNet_VAE takes 80x64 and EmotionNet_VAE2 takes 80x128 
    loss_ = torch.nn.CrossEntropyLoss(reduction='sum')
    encoder = EmotionNet_VAE2(dataset_train='speech1_MelSpec',
                        dataset_val='speech1_MelSpec',
                        device=device,
                        loss_=loss_,
                        load_model=False,
                        beta=config['beta']).to(device=device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    encoder.opt = optimizer

    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lr_lambda=lambda x: x*0.95)

    lr_scheduler = None
    encoder.train_loop(optimizer,
                        lr_scheduler,
                        loss_,
                        batch_size=encoder.config['batch_size'],
                        cpt=None)

with skip_run('skip', 'load_trained_torch_model') as check, check():    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Load the pre-trained model on: {}".format(device))
    
    loss_ = torch.nn.CrossEntropyLoss(reduction='sum')
    encoder = EmotionNet_VAE2(dataset_train='speech1_MelSpec',
                        dataset_val='speech1_MelSpec',
                        device=device,
                        loss_=loss_,
                        load_model=True,
                        beta=config['beta']).to(device=device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    encoder.opt = optimizer
    cpt = 24000 #6100
    encoder.load_model_cpt(cpt=cpt, device=device)
    # encoder.latent_sampling(visualize=True)
    encoder.latent_testing(visualize=True)


######################################################
## Plots of the log-Mel spectrograms of the audio clips
######################################################  
# NOTE:  Only run this if you want to test the generated Mel-Spectrogram
with skip_run('skip', 'verify_original_sample_MelSpec_conversion') as check, check():
        # compare the audio to verify if the conversion is right
        data = dd.io.load((Path(__file__).parents[1] / config['speech1_MelSpec']))
        
        for id in config['actors']:
            for emotion in config['emotions']:
                for intensity in config['intensities']:
                    for repetition in config['repetitions']:
                        # There are no available files for 02 intensity and 01 emotion
                        if (emotion == '01') and (intensity == '02'):
                            continue
                        else: #if emotion == '02':
                            specgram = data['Actor_' + id]['emotion_'+emotion]['intensity_'+intensity]['repete_' + repetition]
                            print('Spectrogram size:', specgram.size())
                            
                            # For inverting procedure
                            if config['use_logMel']:
                                spec = librosa.db_to_power(specgram.data.numpy())
                            else:
                                spec = specgram.data.numpy()
                                
                            audio_clip = librosa.feature.inverse.mel_to_audio(spec,
                                                                sr=config['resampling_rate'],
                                                                n_fft=config['n_fft'],
                                                                hop_length=config['hop_length'],
                                                                win_length=config['win_length'])
                            
                            plt.figure()
                            librosa.display.specshow(librosa.power_to_db(spec),
                                    x_axis='s',
                                    y_axis='mel',
                                    sr=16000,
                                    fmax=8000,cmap='viridis')
                            
                            
                            soundfile.write((str(Path(__file__).parents[1] / 'data/MelSpec_testSamples/actor_') + str(id) + 'emotion_' + str(emotion) + '.wav'),
                                        audio_clip,
                                        samplerate=config['resampling_rate'])  
                                 
            plt.show()
            sys.exit()    

with skip_run('skip', 'plot_constant_len_audio_samples') as check, check():
    # load the dataset
    data = dd.io.load(str(Path(__file__).parents[1] / config['const_MelSpec1']))
    
    actor = 0
    for i in range(len(config['emotions'])):
        input_samples  = data['features'][ data['labels'] == i + 1]
        input_labels   = data['labels'][ data['labels'] == i + 1]
        
        # plot the log-Mel spectrogram        
        plt.figure()
        librosa.display.specshow(input_samples[actor].numpy(),
                x_axis='s',
                y_axis='mel',
                sr=16000,
                fmax=8000,cmap='viridis')
                                                          
        plt.tight_layout()
        
        # save the audio clip
        audio_clip = librosa.feature.inverse.mel_to_audio(librosa.db_to_power(input_samples[0].numpy()),
                                                        sr=config['resampling_rate'],
                                                        n_fft=config['n_fft'],
                                                        hop_length=config['hop_length'],
                                                        win_length=config['win_length'])
        soundfile.write((str(Path(__file__).parents[1] / 'data/MelSpec_testSamples/actor_') + str(actor + 1) + '_emotion_' + str(i + 1) + '.wav'),
                    audio_clip,
                    samplerate=config['resampling_rate'])  

    plt.show()

######################################################
## Operations on the latent space
######################################################                                   
with skip_run('skip', 'Disentangling_by_factorizing') as check, check():
    # load the dataset
    data = dd.io.load(str(Path(__file__).parents[1] / config['const_MelSpec1']))
     
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Load the pre-trained model on: {}".format(device))
    
    vae = EmotionNet_VAE2(dataset_train='speech1_MelSpec',
                        dataset_val='speech1_MelSpec',
                        device=device,
                        loss_=None,
                        load_model=True,
                        beta=config['beta']).to(device=device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    vae.opt = optimizer
    cpt = 6100
    vae.load_model_cpt(cpt=cpt, device=device)
    # vae.latent_sampling(vae.opt, visualize=True)
    
    # calculate the variance using the complete dataset
    input_samples  = data['features'].to(device)

    # get the latent representation of the fixed factor
    mu, logvar = vae.encode(input_samples)
    z = vae.reparameterize(mu, logvar)
    
    # normalize the latent space using the standard deviation of the samples used
    z_std = torch.std(z, dim=0)
        
    
    for i in range(len(config['emotions'])):
        input_samples  = data['features'][ data['labels'] == i + 1].to(device)
        input_labels   = data['labels'][ data['labels'] == i + 1]

        # get the latent representation of the fixed factor
        mu, logvar = vae.encode(input_samples)
        z = vae.reparameterize(mu, logvar)
               
        # normalize the latent space using the standard deviation of complete dataset
        z_norm = torch.div(z, z_std)
        
        # obtain the variance of the normalized embeddings
        val = torch.var(z_norm, dim=0)
        
        # print('Actual emotion: {}, factor with minimum variance: {},'.format(i, torch.argmin(val).cpu().data.numpy()))
        print('Actual emotion: {}, factor with minimum variance: {},'.format(i, val.cpu().data.numpy()))

with skip_run('skip', 'project_latent_space_embeddings_using_UMAP') as check, check():
    # load the dataset
    data = dd.io.load(str(Path(__file__).parents[1] / config['const_MelSpec1']))
    
    # initialize umap
    reducer = umap.UMAP()
     
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Load the pre-trained model on: {}".format(device))
    
    vae = EmotionNet_VAE2(dataset_train='speech1_MelSpec',
                        dataset_val='speech1_MelSpec',
                        device=device,
                        loss_=None,
                        load_model=True,
                        beta=config['beta']).to(device=device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    vae.opt = optimizer
    
    # load the previously trained model (most recently saved model is considered)
    cpt = 24000 #6100
    vae.load_model_cpt(cpt=cpt, device=device)
    
    # calculate the variance using the complete dataset
    input_samples  = data['features'].to(device)
    true_labels    = data['labels']
    # get the latent representation of the fixed factor
    mu, logvar = vae.encode(input_samples)
    z = vae.reparameterize(mu, logvar)
    
    # normalize the latent space using the standard deviation of the samples used
    z_std = torch.std(z, dim=0)        
    
    # normalizing the latent space
    z = torch.div(z, z_std)
    
    embedding = reducer.fit_transform(z.cpu().data.numpy())
    
    plt.scatter(embedding[:, 0], embedding[:, 1], c=true_labels.numpy(), cmap='viridis')
    plt.colorbar() #boundaries=np.arange(8)-0.5).set_ticks(np.arange(8))
    plt.title('UMAP projection of the latent variables', fontsize=15)
    
    plt.show()

with skip_run('skip', 'plot_latent_space_embeddings') as check, check():
    # load the dataset
    data = dd.io.load(str(Path(__file__).parents[1] / config['const_MelSpec1']))
         
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Load the pre-trained model on: {}".format(device))
    
    vae = EmotionNet_VAE2(dataset_train='speech1_MelSpec',
                        dataset_val='speech1_MelSpec',
                        device=device,
                        loss_=None,
                        load_model=True,
                        beta=config['beta']).to(device=device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    vae.opt = optimizer
    
    # load the previously trained model (most recently saved model is considered)
    # the modification has to be made in the file import of line 348 of the emotionet_vae.py
    cpt = 6000 #24000 for model run0 and #6000 for model run2
    vae.load_model_cpt(cpt=cpt, device=device)
    
    # calculate the variance using the complete dataset
    input_samples  = data['features'][:,:, :].to(device)
    true_labels    = data['labels'][:]
    
    print('Total datapoints: {}'.format(true_labels.size()))
    # get the latent representation of the fixed factor
    mu, logvar = vae.encode(input_samples)
    
    std = torch.exp(torch.div(logvar, 2)).cpu().data.numpy()
    
    mu, logvar = mu.cpu().data.numpy(), logvar.cpu().data.numpy()

    cols = np.arange(5, 8) # for 3 embeddings
    # cols = np.arange(1, 9) # for all the embeddings
    
    plt.figure()
    for i in range(mu.shape[0]):
        plt.scatter(mu[i, cols-1], std[i, cols-1], c=cols, cmap= 'viridis')

    plt.colorbar().set_ticks(cols)
    plt.title('latent variables', fontsize=15)
    # plt.tight_layout()
    plt.xlabel('mean', fontsize=15)
    plt.ylabel('standard deviation', fontsize=15)
    plt.show()

#FIXME: not a good idea, because the epsilons are randomly sampled from a normal distribution
# which results in the distortion
with skip_run('skip', 'run_audio_sample_thru_VAE') as check, check():
     # load the dataset
    data = dd.io.load(str(Path(__file__).parents[1] / config['const_MelSpec1']))
     
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Load the pre-trained model on: {}".format(device))
    
    vae = EmotionNet_VAE2(dataset_train='speech1_MelSpec',
                        dataset_val='speech1_MelSpec',
                        device=device,
                        loss_=None,
                        load_model=True,
                        beta=config['beta']).to(device=device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    vae.opt = optimizer
    cpt = 24000
    vae.load_model_cpt(cpt=cpt, device=device)
    # vae.latent_sampling(vae.opt, visualize=True)
    
    # calculate the variance using the complete dataset
    input_samples  = data['features'].to(device)

    # get the output of the VAE
    # mu, logvar = vae.encode(input_samples)
    # z = vae.reparameterize(mu, logvar)
    # x = vae.decode(z)
    x, mu, logvar = vae.forward(input_samples)
    x = x.cpu().data.numpy()
    
    actor = 0
    for i in range(len(config['emotions'])):
        input_samples  = data['features'][ data['labels'] == i + 1]
        input_labels   = data['labels'][ data['labels'] == i + 1]
        
        # plot the log-Mel spectrogram        
        plt.figure()
        librosa.display.specshow(input_samples[actor].numpy(),
                x_axis='s',
                y_axis='mel',
                sr=16000,
                fmax=8000,cmap='viridis')                              
        plt.tight_layout()
        plt.title('Input sample')
        
        plt.figure()
        librosa.display.specshow(x[actor],
                x_axis='s',
                y_axis='mel',
                sr=16000,
                fmax=8000,cmap='viridis')
                                                          
        plt.tight_layout()
        plt.title('Output sample')
        
        # save the audio clip
        audio_clip = librosa.feature.inverse.mel_to_audio(librosa.db_to_power(input_samples[0].numpy()),
                                                        sr=config['resampling_rate'],
                                                        n_fft=config['n_fft'],
                                                        hop_length=config['hop_length'],
                                                        win_length=config['win_length'])
        soundfile.write((str(Path(__file__).parents[1] / 'data/MelSpec_testSamples/actor_') + str(actor + 1) + '_emotion_' + str(i + 1) + '_input.wav'),
                    audio_clip,
                    samplerate=config['resampling_rate'])  
        
        # save the audio clip
        audio_clip = librosa.feature.inverse.mel_to_audio(librosa.db_to_power(x[0]),
                                                        sr=config['resampling_rate'],
                                                        n_fft=config['n_fft'],
                                                        hop_length=config['hop_length'],
                                                        win_length=config['win_length'])
        soundfile.write((str(Path(__file__).parents[1] / 'data/MelSpec_testSamples/actor_') + str(actor + 1) + '_emotion_' + str(i + 1) + '_output.wav'),
                    audio_clip,
                    samplerate=config['resampling_rate'])

    plt.show()



