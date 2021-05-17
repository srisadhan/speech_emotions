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
import soundfile
import webrtcvad
import numpy as np
from scipy.ndimage import binary_dilation
import librosa


def download_dataset(url, config):
    """Downloads the dataset from the provided url and extract the files into the user provided location
    
    Arguments:
        url {string} -- url for downloading the files
        config {dictionary} -- configurations imported from the config.yaml file
    """
    download_to_location = Path(__file__).parents[2] / config['raw_audio_data']

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

def extract_using_torchaudio(config):
    """extract the data from .wav files using torchaudio package
    
    Arguments:
        config {dictionary} -- configurations imported from the config.yaml file
    """
        
    filepath = Path(__file__).parents[2] / config['raw_audio_data']
    VAD = webrtcvad.Vad(config['vad_mode'])

    if config['remove_silences']: # if True func is preprocess else func is empty:returns passed in values
        func = preprocess 
    else:
        func = lambda x,y,z: x

    for statement in config['statements']:
        
        actor_dict = collections.defaultdict()
        for id in config['actors']:
            actorname = 'Actor_' + id
            actor_folder_path = filepath / actorname
            
            emotion_dict = collections.defaultdict()
            for emotion in config['emotions']:

                intensity_dict = collections.defaultdict()
                for intensity in config['intensities']:

                    repetition_dict = collections.defaultdict()
                    for repetition in config['repetitions']:

                        # There are no available files for 02 intensity and 01 emotion
                        if (emotion == '01') and (intensity == '02'):
                            continue
                        else:
                            for file in actor_folder_path.iterdir():
                                sample_name = file.name
                                sample_path = actor_folder_path / sample_name

                                file_info = sample_name.split('-')
                                
                                if (file_info[4] == statement) and (file_info[2] == emotion) and (file_info[3] == intensity) and (file_info[5] == repetition):
                                    # waveform, sample_rate = torchaudio.load(sample_path)

                                    waveform,sample_rate = librosa.load(str(sample_path),mono=True,sr=None)
                                    assert sample_rate == config['sample_rate'], 'sampling rate does not match'
                                    
                                    waveform = torch.Tensor(func(waveform,config,VAD))
                                    repetition_dict['repete_' + repetition] = waveform
                    
                    intensity_dict['intensity_' + intensity] = repetition_dict
                
                emotion_dict['emotion_' + emotion] = intensity_dict
            
            actor_dict[actorname] = emotion_dict
        
        if statement == '01':
            dd.io.save(Path(__file__).parents[2] / config['speech1_data_raw'], actor_dict)
        elif statement == '02':
            dd.io.save(Path(__file__).parents[2] / config['speech2_data_raw'], actor_dict)
        else:
            print('Something is wrong, the file did not return into correct dictionary')


def find_maximum_speech_period(speech_1_path, speech_2_path, statement, config): 
    """find the speech sample with maximum time period and return its length
    
    Arguments:
        speech_1_path {path} -- path to the raw speech_1
        speech_2_path {path} -- path to the raw speech_2
        statement {string}   -- which statement to use
    
    Returns:
        max_speech_period {int} -- maximal length of the time series speech signal
    """

    # Initialize the maximum length of the speech 
    max_speech_period = 0
    
    if statement == '01':
        data = dd.io.load(speech_1_path)
    else:
        data = dd.io.load(speech_2_path)

    for id in config['actors']:
        actorname = 'Actor_' + id
        for emotion in config['emotions']:
            for intensity in config['intensities']:
                for repetition in config['repetitions']:
                    # There are no available files for 02 intensity and 01 emotion
                    if (emotion == '01') and (intensity == '02'):
                        continue
                    else:
                        temp_size = data[actorname]['emotion_'+emotion]['intensity_'+intensity]['repete_' + repetition].numpy().shape[0]
                        if temp_size > max_speech_period:
                            max_speech_period = temp_size
                            # print(actorname, 'emotion_', emotion, 'intensity_', intensity, 'repete_', repetition, ' : ',max_speech_period)
    return max_speech_period


def pad_zerosTo_waveforms(config): 
    """pads zeros to the existing audio signals to make them of constant length
    
    Arguments:
        config {dictionary} -- configurations imported from the config.yaml file

    """
    # path to save the raw data
    speech_1_path = Path(__file__).parents[2] / config['speech1_data_raw']
    speech_2_path = Path(__file__).parents[2] / config['speech2_data_raw']

    # Step 2: pad the audio files with zeros to make them of constant length
    for statement in config['statements']:
        if statement == '01':
            data = dd.io.load(speech_1_path)
        else:
            data = dd.io.load(speech_2_path)

        # calculate the maximum period of the audio signal from all the actors
        max_speech_period = find_maximum_speech_period(speech_1_path, speech_2_path, statement, config)
    
        for id in config['actors']:
            actorname = 'Actor_' + id
            for emotion in config['emotions']:
                for intensity in config['intensities']:
                    for repetition in config['repetitions']:
                        # There are no available files for 02 intensity and 01 emotion
                        if (emotion == '01') and (intensity == '02'):
                            continue
                        else:
                            temp = torch.zeros(max_speech_period)
                            temp_len = data[actorname]['emotion_'+emotion]['intensity_'+intensity]['repete_' + repetition].numpy().shape[0]
                            
                            # sometimes two channels are created for the waveforms and both the channels represent the same data
                            # try to only copy data from one channel
                            
                            temp[:temp_len] = data[actorname]['emotion_'+emotion]['intensity_'+intensity]['repete_' + repetition][:]
                            data[actorname]['emotion_'+emotion]['intensity_'+intensity]['repete_' + repetition] = temp
 
        if statement == '01':
            dd.io.save(Path(__file__).parents[2] / config['speech1_data_refactor'], data)
        elif statement == '02':
            dd.io.save(Path(__file__).parents[2] / config['speech2_data_refactor'], data)
        else:
            print('Something is wrong, the file did not return into correct dictionary')


def club_intensities_as_repetitions(config):
    """Club the audio with varying intensities as single files if voice intensities are not required
    
    Arguments:
        config {dictionary} -- configurations imported from the config.yaml file

    """
    # path to save the raw data
    speech_1_path = Path(__file__).parents[2] / config['speech1_data_refactor']
    speech_2_path = Path(__file__).parents[2] / config['speech2_data_refactor']

    for statement in config['statements']:
        if statement == '01':
            data = dd.io.load(speech_1_path)
        else:
            data = dd.io.load(speech_2_path)

        actor_dict = collections.defaultdict()
        for id in config['actors']:
            actorname = 'Actor_' + id
            
            emotion_dict = collections.defaultdict()
            for emotion in config['emotions']:

                counter = 1
                repetition_dict = collections.defaultdict()
                for intensity in config['intensities']:

                    for repetition in config['repetitions']:

                        # There are no available files for 02 intensity and 01 emotion
                        if (emotion == '01') and (intensity == '02'):
                            continue
                        else:
                            repetition_dict['repete_0' + str(counter)] = data[actorname]['emotion_'+emotion]['intensity_'+intensity]['repete_' + repetition]
                            counter += 1

                emotion_dict['emotion_' + emotion] = repetition_dict
            
            actor_dict[actorname] = emotion_dict
        
        if statement == '01':
            dd.io.save((Path(__file__).parents[2] / config['speech1_no_intensity']), actor_dict)
        elif statement == '02':
            dd.io.save((Path(__file__).parents[2] / config['speech2_no_intensity']), actor_dict)
        else:
            print('Something is wrong, the file did not return into correct dictionary')


def extract_MelSpectrogram(config, intensity_flag, zero_pad_flag, actors=None, data_save_path=None):
    """Extracts MelSpectrogram for the waveforms imported from the audio files
    
    Arguments:
        config {dictionary} -- configurations imported from the config.yaml file
        intensity_flag {bool} -- True if intensities are considered, otherwise False
        actors {list} -- List of actor ids from RAVDESS dataset to extract Melspectrogram
        data_save_path {str} -- Path to store the processed data, defaults to 'None'
    """
    if not actors:
        actors = config['actors']
        
    # path to save the raw data
    if not zero_pad_flag: # don't pad zeros to the data
        speech_1_path = Path(__file__).parents[2] / config['speech1_data_raw']
        speech_2_path = Path(__file__).parents[2] / config['speech2_data_raw']
    elif not intensity_flag: # consider speaker intensity as repetition of statement
        speech_1_path = Path(__file__).parents[2] / config['speech1_no_intensity']
        speech_2_path = Path(__file__).parents[2] / config['speech2_no_intensity']
    else: # 
        speech_1_path = Path(__file__).parents[2] / config['speech1_data_refactor']
        speech_2_path = Path(__file__).parents[2] / config['speech2_data_refactor']

    # max_speech_period = find_maximum_speech_period(speech_1_path, speech_2_path, statement, config)
    
    for statement in config['statements']:
        if statement == '01':
            data = dd.io.load(speech_1_path)
        else:
            data = dd.io.load(speech_2_path)

        for id in actors:
            actorname = 'Actor_' + id
            for emotion in config['emotions']:
                if intensity_flag:
                    for intensity in config['intensities']:
                        for repetition in config['repetitions']:
                            # There are no available files for 02 intensity and 01 emotion
                            if (emotion == '01') and (intensity == '02'):
                                continue
                            else:
                                waveform = data[actorname]['emotion_'+emotion]['intensity_'+intensity]['repete_' + repetition]
                                # specgram = torchaudio.transforms.MelSpectrogram(sample_rate=config['resampling_rate'],
                                                                                # n_mels=config['n_mels'],
                                                                                # n_fft=config['n_fft'],
                                                                                # hop_length=config['hop_length'],
                                                                                # win_length=config['win_length'])(waveform)
                                specgram = librosa.feature.melspectrogram(waveform.data.numpy(), 
                                                                          n_mels=config['n_mels'],
                                                                          sr=config['resampling_rate'],
                                                                          n_fft=config['n_fft'], 
                                                                          hop_length=config['hop_length'], 
                                                                          win_length=config['win_length'])
                                if config['use_logMel']:
                                    spec = librosa.power_to_db(specgram)
                                else:
                                    spec = specgram
                                
                                # Dynamic range compression or clipping is not used here because of log issue
                                # specgram = dynamic_range_compression(torch.Tensor(spec))
                                specgram = torch.Tensor(spec)
                                
                                # print(waveform.data.numpy().shape, specgram.shape)
                                data[actorname]['emotion_'+emotion]['intensity_'+intensity]['repete_' + repetition] = specgram
                else:
                    if (emotion == '01'):
                        repetitions = ['01', '02']
                    else:
                        repetitions = ['01', '02', '03', '04']
                        
                    for repetition in repetitions:                            
                        waveform = data[actorname]['emotion_'+emotion]['repete_' + repetition]
                        # specgram = torchaudio.transforms.MelSpectrogram(sample_rate=config['resampling_rate'],
                        #                                                         n_mels=config['n_mels'],
                        #                                                         n_fft=config['n_fft'],
                        #                                                         hop_length=config['hop_length'],
                        #                                                         win_length=config['win_length'])(waveform)
                        specgram = librosa.feature.melspectrogram(waveform.data.numpy(), 
                                                                  n_mels=config['n_mels'],
                                                                  sr=config['resampling_rate'],
                                                                  n_fft=config['n_fft'], 
                                                                  hop_length=config['hop_length'], 
                                                                  win_length=config['win_length'])
                        if config['use_logMel']:
                            spec = librosa.power_to_db(specgram.data.numpy())
                        else:
                            spec = specgram.data.numpy()
                                    
                        specgram = dynamic_range_compression(torch.Tensor(spec))
                        data[actorname]['emotion_'+emotion]['repete_' + repetition] = specgram
        
        if data_save_path:
            dd.io.save(data_save_path, data)
        else:
            if statement == '01':
                dd.io.save((Path(__file__).parents[2] / config['speech1_MelSpec']), data)
            elif statement == '02':
                dd.io.save((Path(__file__).parents[2] / config['speech2_MelSpec']), data)
            else:
                print('Something is wrong, the file did not return the spectogram correctly')



def normalization(aud, norm_type='peak'):
    """
    Summary:
    
    Args:
    
    Returns:
    
    """
    try:
        assert len(aud) > 0
        if norm_type is 'peak':
            aud = aud / np.max(aud)

        elif norm_type is 'rms':
            dbfs_diff = DBFS - (20 *
                                np.log10(np.sqrt(np.mean(np.square(aud)))))
            if DBFS > 0:
                aud = aud * np.power(10, dbfs_diff / 20)

        return aud
    except AssertionError as e:
        raise AssertionError("Empty audio sig")


def preprocess(aud,config,VAD):
    """
    Summary:
    
    Args:
    
    Returns:
    
    """

    aud = librosa.resample(aud, config['sample_rate'], config['resampling_rate'])
    smoothing_wsize = int(config['smoothing_wsize']*config['resampling_rate']/1000)
    trim_len = len(aud) % smoothing_wsize
    aud = np.append(aud, np.zeros(smoothing_wsize - trim_len))
    assert len(aud) % smoothing_wsize == 0, print(len(aud) % trim_len, aud)

    pcm_16 = np.round((np.iinfo(np.int16).max * aud)).astype(np.int16).tobytes()
    voices = [VAD.is_speech(pcm_16[2 * ix:2 * (ix + smoothing_wsize)],
              sample_rate=config['resampling_rate'])
              for ix in range(0, len(aud), smoothing_wsize)
            ]
    smoothing_mask = np.repeat(
        binary_dilation(voices, np.ones(config['smoothing_length'])), smoothing_wsize)
    aud = aud[smoothing_mask]
    try:
        if len(aud) > 0:
            aud = normalization(aud, norm_type='peak')
        else:
            aud = np.zeros(3*config['resampling_rate'])
        return aud

    except AssertionError as e:
        raise AssertionError("Empty audio sig")


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    #FIXME: Don't use log here, torch.log is log_e not log_10
    # return torch.log(torch.clamp(x, min=clip_val) * C)
    
    #FIXME: don't use clamp while using Log-MelSpectrogram
    # return torch.clamp(x, min=clip_val)
    
    return torch.as_tensor(x)


def constant_shaped_spectrogram(data, config, actors=None):
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
    
    if not actors:
        actors = config['actors']
    
    for id in actors:
        for i, emotion in enumerate(config['emotions']):
            for intensity in config['intensities']:
                for repetition in config['repetitions']:
                    # Emotion 01 - intensity 02 files are not present
                    if (emotion == '01') and (intensity == '02'):
                        continue
                    else:
                        temp = data['Actor_' + id]['emotion_'+emotion]['intensity_'+intensity]['repete_' + repetition].numpy()        
                        slice_ind = abs(temp.shape[1] - config['mel_seg_length']) // 2
                        
                        if temp.shape[1] < config['mel_seg_length']:
                            features.append(librosa.util.fix_length(temp, config['mel_seg_length'], axis=1, mode='edge'))
                        else:
                            features.append(temp[:, slice_ind:slice_ind + config['mel_seg_length']])

                        labels.append(i)
                        speaker_id.append(int(id))
                        
    features = torch.as_tensor(features, dtype=torch.float32)
    labels   = torch.as_tensor(labels, dtype=torch.long)
    speaker_id   = torch.as_tensor(speaker_id)
         
    return features, labels, speaker_id


def pooled_waveform_dataset(data, config, actors=None):
    """Pool all the waveforms and prepare feature and label set

    Parameters
    ----------
    data : dictionary
        dictionary of waveforms extracted for all the emotions from all the actors
    config : dictionary
        configurations mentioned in the config.yml file
        
    Returns
    -------
    features : tensor (n_samples, time)
        a 3d array of stacked spectrograms of emotions
    labels : nd-array (n_samples, 1)
        a 3d array of stacked spectrograms of emotions
    """
    if not actors:
        actors = config['actors']
        
    features = []
    labels   = [] # labels representing the emotion from 0-7
    speaker_id   = [] # labels representing the speaker identity from 01-24, gender information can also be obtained from this 
    
    for id in actors:
        for i, emotion in enumerate(config['emotions']):
            for intensity in config['intensities']:
                for repetition in config['repetitions']:
                    # Emotion 01 - intensity 02 files are not present
                    if (emotion == '01') and (intensity == '02'):
                        continue
                    else:
                        features.append(data['Actor_' + id]['emotion_'+emotion]['intensity_'+intensity]['repete_' + repetition].numpy())
                        
                        labels.append(i)
                        speaker_id.append(int(id))
                        
    # features = torch.as_tensor(features)
    labels   = torch.as_tensor(labels, dtype=torch.long)
    speaker_id   = torch.as_tensor(speaker_id)
         
    return features, labels, speaker_id
