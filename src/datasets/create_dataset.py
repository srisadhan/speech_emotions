import sys
import os
#nopep8
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

import wget 
import zipfile
import collections
import torch 
import torchaudio

import deepdish as dd
from pathlib import Path
import matplotlib.pyplot as plt 
import numpy as np
from src.datasets.data_utils import preprocess_aud, mel_spectrogram, fix_length_spectrogram, return_balanced_data_indices, split_audio_ixs
from src.resemblyzer.voice_encoder import VoiceEncoder
from tqdm import tqdm 

def download_RAVDESS_dataset(url, config):
    """Downloads the dataset from the provided url and extract the files into the user provided location
    
    Arguments:
        url {string} -- url for downloading the files
        config {dictionary} -- configurations imported from the config.yaml file
    """
    download_to_location = Path(__file__).parents[2] / config['raw_audio_data_RAVDESS']

    if not os.path.isfile(os.path.join(download_to_location, 'Audio_Speech_Actors_01-24.zip')):
        print('Downloading the files.........')
        wget.download(url, str(download_to_location))

    for file in download_to_location.iterdir():
        name_split = file.name.split('.')
        if len(name_split) > 1:
            if name_split[1] == 'zip':
                print('Extracting the zip files ........')
                filepath = str(download_to_location / file.name)
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(str(download_to_location))
            else:
                print('No .zip file found')

def load_audio_RAVDESS(config, save_path):
    """extract the waveforms from .wav files using torchaudio package
    
    Arguments:
        config {dictionary} -- configurations imported from the config.yaml file
    """
        
    filepath = Path(__file__).parents[2] / config['raw_audio_data_RAVDESS']

    if config['remove_silences']: # if True func is preprocess else func is empty:returns passed in values
        func = preprocess_aud 
    else:
        func = lambda x,y: (x, y)

    actor_dict = collections.defaultdict()
    for id in tqdm(config['actors']):
        actorname = 'Actor_' + id
        actor_folder_path = filepath / actorname

        state_dict = collections.defaultdict()
        for statement in config['statements']:

            emotion_dict = collections.defaultdict()
            for emotion in config['emotions']:

                intensity_dict = collections.defaultdict()
                for intensity in config['intensities']:

                    repetition_dict = collections.defaultdict()
                    for repetition in config['repetitions']:

                        # There are no files for 02 intensity and 01 emotion
                        if (emotion == '01') and (intensity == '02'):
                            continue
                        else:
                            filename = "03-01-" + emotion + "-" + intensity + "-" + statement + "-" + repetition + "-" + id + ".wav"
                            audio_file = os.path.join(actor_folder_path, filename)

                            # librosa is used to import audio instead of torchaudio
                            # waveform, sample_rate = torchaudio.load(audio_file)
                            # waveform, sample_rate = librosa.load(audio_file,mono=True,sr=None)
                            
                            # preprocess the audio file
                            waveform, _ = func(audio_file)
                            repetition_dict['repete_' + repetition] = waveform

                    intensity_dict['intensity_' + intensity] = repetition_dict
                
                emotion_dict['emotion_' + emotion] = intensity_dict
        
            state_dict['statement_' + statement] = emotion_dict
        
        actor_dict[actorname] = state_dict
        
    dd.io.save(save_path, actor_dict)

def pooled_waveform_dataset_RAVDESS(data, config, actors=None, statements=None, labels_merge=False):
    """Pool all the waveforms and prepare feature and label set

    Parameters
    ----------
    data : dictionary
        dictionary of waveforms extracted for all the emotions from all the actors
    config : dictionary
        configurations mentioned in the config.yml file
        
    Returns
    -------
    features : list (n_samples)
        a 3d array of stacked spectrograms of emotions
    labels : nd-array (n_samples, 1)
        a 3d array of stacked spectrograms of emotions
    """
    if not actors:
        actors = config['actors']
    
    if not statements:
        statements = config['statements'] 
    
    features = []
    labels   = [] # labels representing the emotion from 0-7
    speaker_id   = [] # labels representing the speaker identity from 01-24, gender information can also be obtained from this 

    for id in actors:
        for statement in statements:
            for emotion in config['emotions']:
                for intensity in config['intensities']:
                    for repetition in config['repetitions']:
                        # Emotion 01 - intensity 02 files are not present
                        if (emotion == '01') and (intensity == '02'):
                            continue
                        else:
                            waveform = data['Actor_' + id]['statement_'+statement]['emotion_'+emotion]['intensity_'+intensity]['repete_' + repetition]
                            emotion_label = create_labels_RAVDESS(int(emotion), int(id), config, labels_merge=labels_merge)
                            
                            features.append(waveform)
                            labels.append(emotion_label)
                            speaker_id.append(int(id))
    
    data = {}                        
    data['features']    = features
    data['labels']      = labels
    data['speaker_id']  = speaker_id
         
    return data

def extract_melSpectrogram_RAVDESS(config, save_path, mel_type='simple_40mels', actors=None):
    """Extracts MelSpectrogram for the waveforms imported from the audio files of size (N partial_utterance x 160 x 40)
    
    Arguments:
        config {dictionary} -- configurations imported from the config.yaml file
        mel_type {string} -- Approach to extract Mel-spectrogram. select from two possible strings 'simple_40mels', 'simple_80mels', 'transform'
        actors {list} -- List of actor ids from RAVDESS dataset to extract Melspectrogram
        data_save_path {str} -- Path to store the processed data, defaults to 'None'
    """
    if not actors:
        actors = config['actors']
    
    encoder = VoiceEncoder() 
    # path to save the raw data
    data = dd.io.load(Path(__file__).parents[2] / config['RAVDESS_waveform'])
    Data = collections.defaultdict()

    for id in tqdm(actors):
        actorname = 'Actor_' + id

        statement_mels = collections.defaultdict()
        for statement in config['statements']:
            statement = 'statement_' + statement

            emotion_mels = collections.defaultdict()
            for emotion in config['emotions']:
                repetition_counter = 0

                repetition_mels = collections.defaultdict()
                for intensity in config['intensities']:
                    for repetition in config['repetitions']:
                        # There are no available files for 02 intensity and 01 emotion
                        if (emotion == '01') and (intensity == '02'):
                            continue
                        else:
                            waveform = data[actorname][statement]['emotion_'+emotion]['intensity_'+intensity]['repete_' + repetition]
                            
                            aud_splits, mel_splits = split_audio_ixs(len(waveform))
                            max_aud_length = aud_splits[-1].stop
                            if max_aud_length >= len(waveform):
                                waveform = np.pad(waveform, (0, max_aud_length - len(waveform)), "constant")

                            specgram = mel_spectrogram(waveform, sr=config['resampled_rate'], mel_type=mel_type).T
                            specgram = np.array([specgram[s] for s in mel_splits])

                            # create the embedding using Voice Encoder for each partial utterance
                            embeds = np.array([encoder.embed_utterance(waveform[s]).reshape(1, -1) for s in aud_splits])
                            
                            repetition_counter += 1
                            repetition_mels['repete_' + str(repetition_counter)] = {'mel': specgram,
                                                                                    'embed': embeds}

                emotion_mels['emotion_'+emotion] =  repetition_mels

            statement_mels[statement] = emotion_mels

        Data[actorname]= statement_mels
                            
    dd.io.save(save_path, Data)

def fix_length_melspectrogram_RAVDESS(data, config, actors=None, statements=None, labels_merge=False):
    """Prepare constant length spectrograms of size (n_mels x mel_seg_length)
    for the Variation Auto Encoder

    Parameters
    ----------
    data : dictionary
        dictionary of Melspectrograms extracted for all the emotions from all the actors
    config : dictionary
        configurations mentioned in the config.yml file
    actors : list 
        list of actors from RAVDESS dataset to be used for analysis 
    statements : list 
        statements to use from RAVDESS dataset. By default both the statements are considered ['01', '02']
    Returns
    -------
    features : tensor (n_samples, n_mels, mel_seg_length)
        a 3d array of stacked spectrograms of emotions
    labels : nd-array (n_samples, 1)
        a 3d array of stacked spectrograms of emotions
    """
    
    features = []
    labels   = [] # labels representing the emotion from 0-7
    speaker_id   = [] # labels representing the speaker identity from 01-24, gender information can also be obtained from this 
    speaker_embeds = [] # 256 length speaker embeddings obtained from the VoiceEncoder of Resemblyzer

    if not actors:
        actors = config['actors']
    if not statements:
        statements = config['statements']
        
    for id in actors:
        for statement in statements:
            for emotion in config['emotions']:
                for repetition in config['comb_repets']:
                    # Emotion 01 - intensity 02 files are not present
                    if (emotion == '01') and ((repetition == '3') or (repetition == '4')):
                            continue
                    else:
                        temp = data['Actor_' + id]['statement_'+statement]['emotion_'+emotion]['repete_' + repetition]     
                        
                        specgram = fix_length_spectrogram(temp['mel'], config['mel_seg_length'], append_mode='edge')
                        
                        emotion_label = create_labels_RAVDESS(int(emotion), int(id), config, labels_merge=labels_merge)

                        features.append(specgram)
                        labels.append(emotion_label)
                        speaker_id.append(int(id))
                        speaker_embeds.append(temp['embed'])
    
    balance_idx = return_balanced_data_indices(np.array(labels).reshape(-1, 1))

    data = {}                         
    # data['features'] = torch.as_tensor(features, dtype=torch.float32)
    # data['labels']   = torch.as_tensor(labels, dtype=torch.long)
    # data['speaker_id']   = torch.as_tensor(speaker_id, dtype=torch.long)
    # data['speaker_embeds'] = torch.as_tensor(speaker_embeds, dtype=torch.float32)
    
    data['features'] = np.array(features)[balance_idx]
    data['labels']   = np.array(labels)[balance_idx]
    data['speaker_id']   = np.array(speaker_id)[balance_idx]
    data['speaker_embeds'] = np.array(speaker_embeds)[balance_idx]

    return data

def pool_melspectrogram_RAVDESS(data, config, actors=None, statements=None, labels_merge=False):
    """ Pool the constant length spectrograms extracted using extract_melSpectrogram_RAVDESS of size (n_mels x mel_seg_length)
    for the Variation Auto Encoder

    Parameters
    ----------
    data : dictionary
        dictionary of Melspectrograms extracted for all the emotions from all the actors
    config : dictionary
        configurations mentioned in the config.yml file
    actors : list 
        list of actors from RAVDESS dataset to be used for analysis 
    statements : list 
        statements to use from RAVDESS dataset. By default both the statements are considered ['01', '02']
    Returns
    -------
    features : tensor (n_samples, n_mels, mel_seg_length)
        a 3d array of stacked spectrograms of emotions
    labels : nd-array (n_samples, 1)
        a 3d array of stacked spectrograms of emotions
    """
    
    features = []
    labels   = [] # labels representing the emotion from 0-7
    speaker_id   = [] # labels representing the speaker identity from 01-24, gender information can also be obtained from this 
    speaker_embeds = [] # 256 length speaker embeddings obtained from the VoiceEncoder of Resemblyzer

    if not actors:
        actors = config['actors']
    if not statements:
        statements = config['statements']
        
    for id in actors:
        for statement in statements:
            for emotion in config['emotions']:
                for repetition in config['comb_repets']:
                    # Emotion 01 - intensity 02 files are not present
                    if (emotion == '01') and ((repetition == '3') or (repetition == '4')):
                            continue
                    else:
                        actor_data = data['Actor_' + id]['statement_'+statement]['emotion_'+emotion]['repete_' + repetition]
                        
                        features.append(actor_data['mel'])
                        speaker_embeds.append(actor_data['embed'])
                        
                        emotion_label = create_labels_RAVDESS(int(emotion), int(id), config, labels_merge=labels_merge)
                        emotion_label = np.repeat(emotion_label, actor_data['mel'].shape[0], axis=0)
                        actor_id = np.repeat(int(id), actor_data['mel'].shape[0], axis=0)
 
                        labels.append(emotion_label)
                        speaker_id.append(actor_id)

    features = np.concatenate(features, axis=0)
    speaker_embeds = np.concatenate(speaker_embeds, axis=0)                      
    labels = np.concatenate(labels, axis=0).reshape(-1, 1)
    speaker_id = np.concatenate(speaker_id, axis=0).reshape(-1, 1)
    
    # balance the data using emotion labels
    balance_idx = return_balanced_data_indices(labels)

    data = {}                             
    data['features'] = features[balance_idx]
    data['labels']   = labels[balance_idx]
    data['speaker_id']   = speaker_id[balance_idx]
    data['speaker_embeds'] = speaker_embeds[balance_idx]

    return data


def create_labels_RAVDESS(emotion, id, config, labels_merge=False):
    """ Create labels for the RAVDESS dataset by merging 'Calm' and 'Neutral' emotions 
    resulting in 7 labels for each gender are created yielding 14 labels
    
    if labels_merge=True is passed then additional processing is done to also merge [Happy, Angry, Fear] are combined with [Surprise, Disgust, Sad] 
    are combined respectively resulting in 4 emotion for each gender and in total 8 emotions.
    i.e. 2 with 7, 3 with 5, 4 with 6 are combined
    
    """
    # handle neutral/calm emotion
    if (emotion > 1):
        emotion_label = emotion - 1                          
    else:
        emotion_label = emotion

    if labels_merge:
        if (emotion_label == 7):
            emotion_label = 2
        elif (emotion_label == 6):
            emotion_label = 4
        elif (emotion_label == 5):
            emotion_label = 3
        
        if (id % 2 == 0):
            emotion_label =  (len(config['emotions']) - 4) + emotion_label
    else:
        if (id % 2 == 0):
            emotion_label =  (len(config['emotions']) - 1) + emotion_label

    return emotion_label
