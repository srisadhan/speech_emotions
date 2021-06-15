from torch.optim import optimizer
from umap.umap_ import UMAP
import yaml
import sys

import collections
import os 

#nopep8
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

from src.utils import skip_run
from pathlib import Path
from src.datasets.create_dataset import (download_RAVDESS_dataset, load_audio_RAVDESS,
                                    extract_melSpectrogram_RAVDESS, fix_length_melspectrogram_RAVDESS, 
                                    pooled_waveform_dataset_RAVDESS, pool_melspectrogram_RAVDESS)
from src.datasets.data_utils import gender_from_speaker_id_RAVDESS
from src.datasets.data_loaders import EmotionDataset, EmotionEncoderDataset
from src.models.speaker_encoder import SPEAKER_ENCODER
from src.models.emotionet_ae import EmotionNet_AE
from src.models.emotion_encoder import EmotionEncoder_CONV, EmotionEncoder_LSTM
from src.resemblyzer.voice_encoder import VoiceEncoder
import torch 
import torch.optim as optim

import matplotlib.pyplot as plt 
import deepdish as dd
import argparse
import librosa
import soundfile
import numpy as np
import umap 
from scipy.io.wavfile import write
from src.resemblyzer import audio
import librosa
from tqdm import tqdm

# The configuration file
config = yaml.load(open('src/config.yaml'), Loader=yaml.SafeLoader)

# Create the required folders
os.makedirs(str(Path(__file__).parents[1] / 'data'), exist_ok=True)
os.makedirs(str(Path(__file__).parents[1] / 'data/raw'), exist_ok=True)
os.makedirs(str(Path(__file__).parents[1] / 'data/interim'), exist_ok=True)
os.makedirs(str(Path(__file__).parents[1] / 'data/processed'), exist_ok=True)

######################################################
## RAVDESS dataset
######################################################
# download data from web and place it in the desired location
with skip_run('skip', 'download_RAVDESS_data') as check, check():
    url = config['RAVDESS_url']
    download_RAVDESS_dataset(url, config)

# import the data from .wav files and create a dictionary
with skip_run('skip', 'import audio files') as check, check():
    save_path = Path(__file__).parents[1] / config['RAVDESS_waveform']
    load_audio_RAVDESS(config, save_path)

with skip_run('skip', 'extract MelSpec and save') as check, check():
    save_path = Path(__file__).parents[1] / config['RAVDESS_simple_40melspec']
    extract_melSpectrogram_RAVDESS(config, save_path, mel_type='simple_40mels')
    
    # save_path = Path(__file__).parents[1] / config['RAVDESS_simple_80melspec']
    # extract_melSpectrogram_RAVDESS(config, save_path, mel_type='simple_80mels')
    
    # save_path = Path(__file__).parents[1] / config['RAVDESS_transform_80melspec']
    # extract_melSpectrogram_RAVDESS(config, save_path, mel_type='transform')

with skip_run('skip', 'Make fix length mel spectrograms') as check, check():
    # ---------------fix length spectrograms for 40 mels 
    data = dd.io.load((Path(__file__).parents[1] / config['RAVDESS_simple_40melspec']))
    Data = pool_melspectrogram_RAVDESS(data, config, actors=config['train_actors'])
    dd.io.save(str(Path(__file__).parents[1] / config['const_40mel_simple']), Data)          

    # ---------------fix length spectrograms for 80 mels 
    # data = dd.io.load((Path(__file__).parents[1] / config['RAVDESS_simple_80melspec']))
    # Data = pool_melspectrogram_RAVDESS(data, config, actors=config['train_actors'])
    # dd.io.save(str(Path(__file__).parents[1] / config['const_80mel_simple']), Data) 
    
    # # ---------------fix length spectrograms for 80 mel STFT 
    # data = dd.io.load((Path(__file__).parents[1] / config['RAVDESS_transform_80melspec']))
    # Data = pool_melspectrogram_RAVDESS(data, config, actors=config['train_actors'])
    # dd.io.save(str(Path(__file__).parents[1] / config['const_80mel_transform']), Data) 
    
######################################################
## Speaker Encoder - Chaitanya's trained model
###################################################### 
with skip_run('skip', 'Project the Speaker Embeddings to a Manifold') as check, check():
    data = dd.io.load(Path(__file__).parents[1] / config['RAVDESS_waveform'])
    data = pooled_waveform_dataset_RAVDESS(data, config, actors=config['train_actors'], statements=['01'])

    reducer = umap.UMAP()
    
    model = SPEAKER_ENCODER(load_model=True,
                            epoch=0,
                            device=torch.device('cpu'),
                            loss_=None, mode='validate')
    
    model.load_model_cpt(cpt=13, device=torch.device('cpu'))

    model.eval()
    
    speaker_embedding = []
    
    for i in range(len(data['features'])):
        embeds, _ = model.embed(data['features'][i], group=True)
        speaker_embedding.append(embeds.reshape(1, -1))
    
    speaker_embedding = np.concatenate(speaker_embedding, axis=0)
    manifold_embedding = reducer.fit_transform(speaker_embedding)
    
    gender_id = gender_from_speaker_id_RAVDESS(config, data['speaker_id'])
    
    female_ind = np.multiply(np.arange(0, gender_id.shape[0]), gender_id.tolist())
    male_ind   = np.multiply(np.arange(0, gender_id.shape[0]), np.subtract(np.ones(gender_id.shape, dtype=np.int), gender_id, dtype=np.int).tolist())

    plt.figure()
    plt.scatter(manifold_embedding[:, 0], manifold_embedding[:, 1], c=gender_id, cmap='viridis')
    plt.colorbar()
    plt.title('Male and Female Speaker embeddings on the UMAP manifold')
    
    plt.figure()
    plt.scatter(manifold_embedding[male_ind, 0], manifold_embedding[male_ind, 1], c=data['speaker_id'][male_ind], cmap='viridis')
    plt.colorbar()
    plt.title('Male Speaker embeddings on the UMAP manifold')
    
    plt.figure()
    plt.scatter(manifold_embedding[female_ind, 0], manifold_embedding[female_ind, 1], c=data['speaker_id'][female_ind], cmap='viridis')
    plt.colorbar()
    plt.title('Female Speaker embeddings on the UMAP manifold')
    
    
######################################################
## Voice Encoder - from Resemblyzer project
###################################################### 
with skip_run('skip', 'Project the Voice Embeddings to a Manifold') as check, check():
    reducer = umap.UMAP()
    encoder = VoiceEncoder()

    filename = 'RAVDESS_simple_40melspec'
    data = dd.io.load((Path(__file__).parents[1] / config[filename]))
    Data = pool_melspectrogram_RAVDESS(data, config, actors=config['train_actors'])

    speaker_embedding = Data['speaker_embeds'].reshape(-1, 256)
    manifold_embedding = reducer.fit_transform(speaker_embedding)
    speaker_id = Data['speaker_id']


    gender_id = gender_from_speaker_id_RAVDESS(config, speaker_id)
    female_ind = np.multiply(np.arange(0, gender_id.shape[0]), gender_id.tolist())
    male_ind   = np.multiply(np.arange(0, gender_id.shape[0]), np.subtract(np.ones(gender_id.shape, dtype=np.int), gender_id, dtype=np.int).tolist())

    plt.figure()
    plt.scatter(manifold_embedding[:, 0], manifold_embedding[:, 1], c=gender_id, cmap='viridis')
    plt.colorbar()
    plt.title('Male and Female Speaker embeddings on the UMAP manifold')
    plt.savefig('Male_Female_embeds.png')

    plt.figure()
    plt.scatter(manifold_embedding[male_ind, 0], manifold_embedding[male_ind, 1], c=speaker_id[male_ind], cmap='viridis')
    plt.colorbar()
    plt.title('Male Speaker embeddings on the UMAP manifold')
    plt.savefig('Male_embeds.png')

    plt.figure()
    plt.scatter(manifold_embedding[female_ind, 0], manifold_embedding[female_ind, 1], c=speaker_id[female_ind], cmap='viridis')
    plt.colorbar()
    plt.title('Female Speaker embeddings on the UMAP manifold')
    plt.savefig('Female_embeds.png')
    
######################################################
## Train EmotionNet
###################################################### 
with skip_run('skip', 'Train the Emotion autoencoder with specified mel spec') as check, check():
    # Exploratory study of using different mels with and w/o speaker embeddings
    N_MELS = [80, 80, 40, 40]
    MEL_TYPE = ['transform', 'transform', 'simple', 'simple']
    SPEAKER_EMBED = [False, True, False, True]
        
    for i in range(len(N_MELS)):        
        # please provide the number of mels to be used 
        n_mels = N_MELS[i]
        mel_type = MEL_TYPE[i]
        use_speaker_embeds = SPEAKER_EMBED[i]
        
        print('Train the model with Mels: {}, Type: {}, Speaker embed used:{}'.format(n_mels, mel_type, use_speaker_embeds))
    
        filename = 'RAVDESS_' + mel_type + '_' + str(n_mels) + 'melspec'
        data = dd.io.load((Path(__file__).parents[1] / config[filename]))
        
        Data = fix_length_melspectrogram_RAVDESS(data, config, actors=config['train_actors'])
        
        dataset = EmotionDataset(Data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                
        model = EmotionNet_AE(n_mels=n_mels, device=device, use_speaker_embeds=use_speaker_embeds)
        model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        scheduler = None
        
        model.train_model(dataloader, optimizer, device, lr_scheduler=scheduler)
        
        test_data = fix_length_melspectrogram_RAVDESS(data, config, actors=config['test_actors'])
        
        test_dataset = EmotionDataset(test_data)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        model.validate_model(test_dataloader)
    
with skip_run('skip', 'Train the EmotionEncoder convolution model') as check, check():
    N_MELS = [40, 40] #, 80, 80]
    MEL_TYPE = ['simple', 'simple']#, 'transform', 'transform']
    SPEAKER_EMBED = [False, True]#, False, True]
    utterances_per_emotion = 8
    emotions_per_batch = 14

    for i in range(len(N_MELS)):        
        # please provide the number of mels to be used 
        n_mels = N_MELS[i]
        mel_type = MEL_TYPE[i]
        use_speaker_embeds = SPEAKER_EMBED[i]
        
        print('Train the model with Mels: {}, Type: {}, Speaker embed used:{}'.format(n_mels, mel_type, use_speaker_embeds))
    
        filename = 'RAVDESS_' + mel_type + '_' + str(n_mels) + 'melspec'
        data = dd.io.load((Path(__file__).parents[1] / config[filename]))
        
        Data = fix_length_melspectrogram_RAVDESS(data, config, actors=config['train_actors'])

        train_dataset = EmotionEncoderDataset(Data, emotions_per_batch=emotions_per_batch, mode='train')
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=utterances_per_emotion, shuffle=False)
        
        Data = fix_length_melspectrogram_RAVDESS(data, config, actors=config['valid_actors'])
        valid_dataset = EmotionEncoderDataset(Data, emotions_per_batch=emotions_per_batch, mode='train')
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=utterances_per_emotion, shuffle=False)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                
        model = EmotionEncoder_CONV(n_mels=n_mels, device=device, use_speaker_embeds=use_speaker_embeds)
        model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        scheduler = None
        
        model.train_model(train_dataloader,
                   valid_dataloader,
                   emotions_per_batch=emotions_per_batch, 
                   optimizer=optimizer,
                   device=device,
                   lr_scheduler=scheduler)
        
        test_data = fix_length_melspectrogram_RAVDESS(data, config, actors=config['test_actors'])
        
        test_dataset = EmotionEncoderDataset(test_data, emotions_per_batch=emotions_per_batch, mode='train')
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        model.validate_model(test_dataloader)

with skip_run('skip', 'Train the EmotionEncoder LSTM model') as check, check():        
    N_MELS = [40, 40] #, 80, 80]
    MEL_TYPE = ['simple', 'simple']#, 'transform', 'transform']
    SPEAKER_EMBED = [False, True]#, False, True]
    utterances_per_emotion = 8
    emotions_per_batch = 8
    labels_merge=True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for i in range(len(N_MELS)):        
        # please provide the number of mels to be used 
        n_mels = N_MELS[i]
        mel_type = MEL_TYPE[i]
        use_speaker_embeds = SPEAKER_EMBED[i]
        
        print('Train the model with Mels: {}, Type: {}, Speaker embed used:{}'.format(n_mels, mel_type, use_speaker_embeds))
    
        filename = 'RAVDESS_' + mel_type + '_' + str(n_mels) + 'melspec'
        data = dd.io.load((Path(__file__).parents[1] / config[filename]))
        
        Data = pool_melspectrogram_RAVDESS(data, config, actors=config['train_actors'], labels_merge=labels_merge)
        train_dataset = EmotionEncoderDataset(Data, emotions_per_batch=emotions_per_batch, utterances_per_emotion=utterances_per_emotion, mode='train')
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        
        Data = pool_melspectrogram_RAVDESS(data, config, actors=config['valid_actors'], labels_merge=labels_merge)
        valid_dataset = EmotionEncoderDataset(Data, emotions_per_batch=emotions_per_batch, utterances_per_emotion=utterances_per_emotion, mode='train')
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)
                
        model = EmotionEncoder_LSTM(n_mels=n_mels, device=device, use_speaker_embeds=use_speaker_embeds)
        model.to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
        # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        scheduler = None
        
        model.train_model(train_dataloader,
                   valid_dataloader,
                   emotions_per_batch=emotions_per_batch, 
                   optimizer=optimizer,
                   device=device,
                   lr_scheduler=scheduler)
        
        test_data = pool_melspectrogram_RAVDESS(data, config, actors=config['test_actors'], labels_merge=labels_merge) 
        test_dataset = EmotionEncoderDataset(test_data, emotions_per_batch=emotions_per_batch, utterances_per_emotion=utterances_per_emotion, mode='train')
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        model.validate_model(test_dataloader)

with skip_run('run', 'Validate the EmotionEncoder LSTM model') as check, check():        
    N_MELS = [40, 40] #, 80, 80]
    MEL_TYPE = ['simple', 'simple']#, 'transform', 'transform']
    SPEAKER_EMBED = [True, False]#, False, True]
    LABELS_MERGE = [False, True]    
    
    for labels_merge in LABELS_MERGE:
        if labels_merge:
            n_emotions = 8
        else:
            n_emotions = 14
            
        data = dd.io.load(Path(__file__).parents[1] / config['RAVDESS_waveform'])   
        wav_data = pooled_waveform_dataset_RAVDESS(data, config, labels_merge=labels_merge, actors=config['train_actors'])
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for i in range(len(N_MELS)):        
            # please provide the number of mels to be used 
            n_mels = N_MELS[i]
            mel_type = MEL_TYPE[i]
            use_speaker_embeds = SPEAKER_EMBED[i]
            
            print('Validate the model with Mels: {}, Type: {}, Speaker embed used:{}'.format(n_mels, mel_type, use_speaker_embeds))
        
            folder_name = 'mel_' + str(n_mels) + '_spk_' + str(use_speaker_embeds) + '_emo_' + str(n_emotions)
            file_path = Path(__file__).parents[1] / config['model_save_dir'] / 'EmotionEncoder_LSTM' / folder_name
            
            # for file in file_path.iterdir():
            #     name_split = file.name.split('_')
            #     epoch = str(name_split[-1].split('.')[0])
            #     print(epoch)
                
            file_name = 'EmotionEncoder_LSTM_Epoch_5000.pt'
            checkpoint = torch.load(os.path.join(config['model_save_dir'], 'EmotionEncoder_LSTM', folder_name, file_name))
            
            model = EmotionEncoder_LSTM(n_mels=n_mels, device=device, use_speaker_embeds=use_speaker_embeds)
            model.load_model_from_dict(checkpoint)
            model.to(device)
            
            emotion_embeds, emotion_labels = [], []
            for i, wav in tqdm(enumerate(wav_data['features'])):
                emotion_embeds.append(model.embed_emotion(wav).reshape(1, -1))
                emotion_labels.append(wav_data['labels'][i])

            emotion_embeds = np.concatenate(emotion_embeds, axis=0)
            
            reducer = umap.UMAP()
            umap_embeds = reducer.fit_transform(emotion_embeds)
            
            plt.figure()
            plt.scatter(umap_embeds[:, 0], umap_embeds[:, 1], c=emotion_labels, cmap='viridis')
            plt.colorbar()
            plt.title('Speaker: ' + str(use_speaker_embeds) + ' Labels: ' + str(n_emotions))
            plt.savefig('Speaker_' + str(use_speaker_embeds) + '_Labels_' + str(n_emotions) + '.png')
            # filename = 'RAVDESS_' + mel_type + '_' + str(n_mels) + 'melspec'
            # data = dd.io.load((Path(__file__).parents[1] / config[filename]))
            
            # Data = pool_melspectrogram_RAVDESS(data, config, actors=config['train_actors'], labels_merge=labels_merge)
            # train_dataset = EmotionEncoderDataset(Data, emotions_per_batch=emotions_per_batch, utterances_per_emotion=utterances_per_emotion, mode='train')
            # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
            # model.validate_model(train_dataloader)
            
            # Data = pool_melspectrogram_RAVDESS(data, config, actors=config['valid_actors'], labels_merge=labels_merge)
            # valid_dataset = EmotionEncoderDataset(Data, emotions_per_batch=emotions_per_batch, utterances_per_emotion=utterances_per_emotion, mode='train')
            # valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)
            # model.validate_model(valid_dataloader)
            
            # test_data = pool_melspectrogram_RAVDESS(data, config, actors=config['test_actors'], labels_merge=labels_merge)
            # test_dataset = EmotionEncoderDataset(test_data, emotions_per_batch=emotions_per_batch, utterances_per_emotion=utterances_per_emotion, mode='train')
            # test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
            # model.validate_model(test_dataloader)

######################################################
## TACOTRON
###################################################### 
with skip_run('skip', 'test pytorch TACOTRON and WAVEGLOW models') as check, check():
    # load tacotron2
    tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
    tacotron2.to('cuda')
    tacotron2.eval()
    
    # load waveglow
    waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.to('cuda')
    waveglow.eval()
    
    text = "Kids are talking by the door"
    sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)
    
    # run the models
    with torch.no_grad():
        _, mel, _, _ = tacotron2.infer(sequence)
        audio = waveglow.infer(mel)
    audio_numpy = audio[0].data.cpu().numpy()
    rate = 22050
    plt.plot(audio_numpy)
    
    write("audio.wav",  rate, audio_numpy)

with skip_run('skip', 'test WAVEGLOW model with simple and transform log-mels') as check, check():
    tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
    tacotron2.to('cuda')
    tacotron2.eval()   
    text = "Kids are talking by the door"
    sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)
    
    # run the models
    with torch.no_grad():
        _, mel, _, _ = tacotron2.infer(sequence)
                                      
    simple_40logmel_path = Path(__file__).parents[1] / config['RAVDESS_simple_40melspec']
    simple_80logmel_path = Path(__file__).parents[1] / config['RAVDESS_simple_80melspec']
    transform_logmel_path = Path(__file__).parents[1] / config['RAVDESS_transform_80melspec']
    
    data_simple_40 = dd.io.load(simple_40logmel_path)
    data_simple_80 = dd.io.load(simple_80logmel_path)
    data_transform = dd.io.load(transform_logmel_path)

    mel1 = data_simple_40['Actor_06']['statement_01']['emotion_08']['repete_4']
    mel2 = data_simple_80['Actor_06']['statement_01']['emotion_08']['repete_4']
    mel3 = data_transform['Actor_06']['statement_01']['emotion_08']['repete_4']
    
    
    plt.figure()
    librosa.display.specshow(mel.squeeze().cpu().numpy())
    plt.colorbar()
    plt.figure()
    librosa.display.specshow(mel1)
    plt.colorbar()
    plt.figure()
    librosa.display.specshow(mel2)
    plt.colorbar()
    plt.figure()
    librosa.display.specshow(mel3)
    plt.colorbar()
    plt.savefig('temp.png')

        
    # load waveglow
    waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow.to('cuda')
    waveglow.eval()
    
    waveglow_rate = 22050
    librosa_rate  = 16000
    
    # ----------- TACOTRON 80 Mels
    # run the models
    with torch.no_grad():
        audio = waveglow.infer(torch.tensor(mel3.reshape(1, 80, -1), dtype=torch.float32).to('cuda'))
    audio_numpy = audio[0].data.cpu().numpy()
    soundfile.write("audio_torch_transform.wav", audio_numpy, waveglow_rate)
    
    mel3 = librosa.db_to_power(mel3)
    y = librosa.feature.inverse.mel_to_audio(mel3,
                                            sr=librosa_rate,
                                            n_fft=1024,
                                            hop_length=256,
                                            win_length=1024)

    soundfile.write("audio_librosa_transform.wav", y, librosa_rate)
    
    
    # ----------- 40 Mels
    mel1 = librosa.db_to_power(mel1)
    y = librosa.feature.inverse.mel_to_audio(mel1,
                                            sr=librosa_rate,
                                            n_fft=400,
                                            hop_length=160,
                                            win_length=400)
    
    soundfile.write("audio_librosa_simple_40.wav", y, librosa_rate)
    
    # ----------- 80 Mels
    # run the models
    with torch.no_grad():
        audio = waveglow.infer(torch.tensor(mel2.reshape(1, 80, -1), dtype=torch.float32).to('cuda'))
    audio_numpy = audio[0].data.cpu().numpy()

    soundfile.write("audio_torch_simple_80.wav", audio_numpy, waveglow_rate)
    
    mel2 = librosa.db_to_power(mel2)
    y = librosa.feature.inverse.mel_to_audio(mel2,
                                            sr=librosa_rate,
                                            n_fft=1024,
                                            hop_length=256,
                                            win_length=1024)
    soundfile.write("audio_librosa_simple_80.wav", y, librosa_rate)
    # NVIDIA's waveglow
    # waveglow_nvidia = torch.load(config['waveglow_model'], map_location=torch.device('cuda'))['model']
    # waveglow_nvidia.eval()
    # with torch.no_grad():
    #     audio = waveglow_nvidia.infer(torch.tensor(mel.reshape(1, 80, -1), dtype=torch.float32).to('cuda'))
    # audio_numpy = audio[0].data.cpu().numpy()
    # rate = 16000
    # soundfile.write("audio_nvidia.wav", audio_numpy, waveglow_rate)
    
plt.show()
    
    
    