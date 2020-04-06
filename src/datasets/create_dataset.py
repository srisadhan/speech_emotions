import wget 
import zipfile
import collections
import torch 
import torchaudio

import deepdish as dd
from pathlib import Path
import matplotlib.pyplot as plt 

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
                                    waveform, sample_rate = torchaudio.load(sample_path)
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
                            temp_size = data[actorname]['emotion_'+emotion]['intensity_'+intensity]['repete_' + repetition].numpy().shape[1]
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
                            temp = torch.zeros(1, max_speech_period, dtype=torch.float64)
                            temp_len = data[actorname]['emotion_'+emotion]['intensity_'+intensity]['repete_' + repetition].numpy().shape[1]
                            
                            # sometimes two channels are created for the waveforms and both the channels represent the same data
                            # try to only copy data from one channel
                            temp[0, :temp_len] = data[actorname]['emotion_'+emotion]['intensity_'+intensity]['repete_' + repetition][0,:]
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


def extract_MelSpectrogram(config, intensity_flag):
    """Extracts MelSpectrogram for the waveforms imported from the audio files
    
    Arguments:
        config {dictionary} -- configurations imported from the config.yaml file
        intensity_flag {bool} -- True if intensities are considered, otherwise False
    """
    # path to save the raw data
    if intensity_flag:
        speech_1_path = Path(__file__).parents[2] / config['speech1_no_intensity']
        speech_2_path = Path(__file__).parents[2] / config['speech2_no_intensity']
    else:
        speech_1_path = Path(__file__).parents[2] / config['speech1_data_refactor']
        speech_2_path = Path(__file__).parents[2] / config['speech2_data_refactor']
        
    
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
                                specgram = torchaudio.transforms.MelSpectrogram()(waveform)
                                data[actorname]['emotion_'+emotion]['intensity_'+intensity]['repete_' + repetition] = specgram
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