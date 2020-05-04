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


def find_maximum_speech_period(speech_1_path, speech_2_path, config): 
    """find the speech sample with maximum time period and return its length
    
    Arguments:
        speech_1_path {path} -- path to the raw speech_1
        speech_2_path {path} -- path to the raw speech_2
    
    Returns:
        max_speech_period {int} -- maximal length of the time series speech signal
    """

    # Initialize the maximum length of the speech 
    max_speech_period = 0
    
    for statement in config['statements']:
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
    return max_speech_period


def pad_zerosTo_waveforms(config): 
    """pads zeros to the existing audio signals to make them of constant length
    
    Arguments:
        config {dictionary} -- configurations imported from the config.yaml file

    """
    # path to save the raw data
    speech_1_path = Path(__file__).parents[2] / config['speech1_data_raw']
    speech_2_path = Path(__file__).parents[2] / config['speech2_data_raw']

    # Step 1: calculate the maximum period of the audio signal from all the actors
    max_speech_period = find_maximum_speech_period(speech_1_path, speech_2_path, config)
    
    # Step 2: pad the audio files with zeros to make them of constant length
    for statement in config['statements']:
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
                            temp = torch.zeros(1, max_speech_period)
                            temp_len = data[actorname]['emotion_'+emotion]['intensity_'+intensity]['repete_' + repetition].numpy().shape[0]
                            
                            # sometimes two channels are created for the waveforms and both the channels represent the same data
                            # try to only copy data from one channel
                            
                            temp[0, :temp_len] = data[actorname]['emotion_'+emotion]['intensity_'+intensity]['repete_' + repetition][:]
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


def extract_MelSpectrogram(config, intensity_flag, zero_pad_flag):
    """Extracts MelSpectrogram for the waveforms imported from the audio files
    
    Arguments:
        config {dictionary} -- configurations imported from the config.yaml file
        intensity_flag {bool} -- True if intensities are considered, otherwise False
    """
    # path to save the raw data
    if not zero_pad_flag:
        speech_1_path = Path(__file__).parents[2] / config['speech1_data_raw']
        speech_2_path = Path(__file__).parents[2] / config['speech2_data_raw']
    elif not intensity_flag:
        speech_1_path = Path(__file__).parents[2] / config['speech1_no_intensity']
        speech_2_path = Path(__file__).parents[2] / config['speech2_no_intensity']
    else:
        speech_1_path = Path(__file__).parents[2] / config['speech1_data_refactor']
        speech_2_path = Path(__file__).parents[2] / config['speech2_data_refactor']

    max_speech_period = find_maximum_speech_period(speech_1_path, speech_2_path, config)
    
    for statement in config['statements']:
        if statement == '01':
            data = dd.io.load(speech_1_path)
        else:
            data = dd.io.load(speech_2_path)

        for id in config['actors']:
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
                                specgram = torchaudio.transforms.MelSpectrogram(n_mels=config['n_mels'],
                                                                                n_fft=config['n_fft'],
                                                                                hop_length=config['hop_length'],
                                                                                win_length=config['win_length'])(waveform)
                                specgram = dynamic_range_compression(specgram)
                                data[actorname]['emotion_'+emotion]['intensity_'+intensity]['repete_' + repetition] = specgram
                                torchaudio.transforms.MelSpectrogram()
                else:
                    for repetition in ['repete_01', 'repete_02', 'repete_03', 'repete_04']:                            
                        waveform = data[actorname]['emotion_'+emotion]['repete_' + repetition]
                        specgram = torchaudio.transforms.MelSpectrogram()(waveform)
                        data[actorname]['emotion_'+emotion]['repete_' + repetition] = specgram
        
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

    pcm_16 = np.round(
        (np.iinfo(np.int16).max * aud)).astype(np.int16).tobytes()
    voices = [
        VAD.is_speech(pcm_16[2 * ix:2 * (ix + smoothing_wsize)],
                      sample_rate=config['resampling_rate'])
        for ix in range(0, len(aud), smoothing_wsize)
    ]
    smoothing_mask = np.repeat(
        binary_dilation(voices, np.ones(config['smoothing_length'])), smoothing_wsize)
    aud = aud[smoothing_mask]
    try:
        aud = normalization(aud, norm_type='peak')
        return aud

    except AssertionError as e:
        raise AssertionError("Empty audio sig")


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)