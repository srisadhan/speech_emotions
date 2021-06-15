import matplotlib
from random import randint, sample
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import h5py
import warnings
from tqdm import tqdm
from colorama import Fore
import pathlib
from scipy.ndimage import binary_dilation
import soundfile
import webrtcvad
import numpy as np
from librosa.display import specshow
import librosa
from collections import Counter
import pandas as pd
from yaml import safe_load
import sys
import os
from sklearn.metrics import confusion_matrix
from itertools import product
import json
from librosa.filters import mel as librosa_mel_fn
from imblearn.under_sampling import RandomUnderSampler

# nopep8
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())
from src.datasets.stft import STFT  # nopep8


warnings.filterwarnings("ignore")

with open('src/config.yaml', 'r') as f:
    config = safe_load(f.read())

RAW_DATA_DIR = config['raw_audio_data_RAVDESS']
PROC_DATA_DIR = config['proc_data_dir']
INTERIM_DATA_DIR = config['interim_data_dir']
MODEL_SAVE_DIR = config['model_save_dir']
VIS_DIR = config['vis_dir']
RUNS_DIR = config['runs_dir']

SAMPLING_RATE = config['resampled_rate']

N_FFT = config['n_fft']
H_L = config['hop_length']
STEP_SIZE_EM = int((SAMPLING_RATE/16)/H_L)
MEL_CHANNELS = config['n_mels']
SMOOTHING_LENGTH = config['smoothing_length']
SMOOTHING_WSIZE = config['smoothing_wsize']
WINDOW_STEP = config['window_step']
DBFS = config['dbfs']
SMOOTHING_WSIZE = int(SMOOTHING_WSIZE * SAMPLING_RATE / 1000)

dirs_ = set([globals()[d] for d in globals() if d.__contains__('DIR')] +
            [config[d] for d in config if d.__contains__('DIR')])

VAD = webrtcvad.Vad(mode=config['vad_mode'])


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.mel_channels = mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, mel_channels, T)
        """
        # assert(torch.min(y.data) >= -1)
        # assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output


STFT = TacotronSTFT(
    config['n_fft_tron'],   config['hop_length_tron'],    config['win_length_tron'],
    config['n_mels_tron'],  SAMPLING_RATE, config['mel_fmin_tron'],
    config['mel_fmax_tron']
)
# To work with 40 mels
# STFT = TacotronSTFT(
#     config['n_fft'],   config['hop_length'],    config['win_length'],
#     config['n_mels'],  SAMPLING_RATE, config['mel_fmin_tron'],
#     config['mel_fmax_tron']
# )

def structure(dirs=[]):
    """
    Create necessary directory structure

    Args:

    Returns:

    """
    dirs_reqd = set(list(dirs_) + list(dirs))
    for data_dir in dirs_reqd:
        if not pathlib.Path.exists(pathlib.Path(data_dir)):
            os.makedirs(data_dir)


def normalization(aud, norm_type='peak'):
    """
    Normalize the given audio clip

    Args:

    Returns:

    """
    try:
        assert len(aud) > 0
        if norm_type == "peak":
            aud = aud / np.max(aud)

        elif norm_type == "rms":
            dbfs_diff = DBFS - (20 *
                                np.log10(np.sqrt(np.mean(np.square(aud)))))
            if DBFS > 0:
                aud = aud * np.power(10, dbfs_diff / 20)

        return aud
    except AssertionError as e:
        raise AssertionError("Empty audio sig")


def detect_voices(aud, sr=44100):
    """
    Detect the presence and absence of voices in an array of audio 

    Args:

    Returns:

    """
    pcm_16 = np.round(
        (np.iinfo(np.int16).max * aud)).astype(np.int16).tobytes()
    voices = [
        VAD.is_speech(pcm_16[2 * ix:2 * (ix + SMOOTHING_WSIZE)],
                      sample_rate=SAMPLING_RATE)
        for ix in range(0, len(aud), SMOOTHING_WSIZE)
    ]
    return voices


def preprocess_aud(aud_input, sr=44100):
    """
    Resample, Preprocess and return the provided audio

    Args:

    Returns:

    """
    if isinstance(aud_input, list) or isinstance(aud_input, np.ndarray):
        aud = np.array(aud_input)
    else:
        fname = aud_input
        aud, sr = librosa.load(fname, sr=None)
    if sr != SAMPLING_RATE:
        aud = librosa.resample(aud, sr, SAMPLING_RATE)
        
    try:
        aud = normalization(aud, norm_type='peak')
    except AssertionError as e:
        print(AssertionError("Empty audio sig"))

    trim_len = len(aud) % SMOOTHING_WSIZE
    aud = np.append(aud, np.zeros(SMOOTHING_WSIZE - trim_len))

    assert len(aud) % SMOOTHING_WSIZE == 0, print(len(aud) % trim_len, aud)

    # remove silences from the audio clip
    voices = detect_voices(aud, SAMPLING_RATE)

    smoothing_mask = np.repeat(
        binary_dilation(voices, np.ones(SMOOTHING_LENGTH)), SMOOTHING_WSIZE)
    aud = aud[smoothing_mask]

    try:
        aud = normalization(aud, norm_type='peak')
        return aud, SAMPLING_RATE

    except AssertionError as e:
        print(AssertionError("Empty audio sig"))
        return aud, SAMPLING_RATE
        # exit()


def mel_spectrogram(aud, sr=16000, mel_type='simple_40mels'):
    """
    Summary:

    Args:

    Returns:

    """
    if mel_type == 'simple_40mels':
        mel = librosa.feature.melspectrogram(aud,
                                             sr=sr,
                                             n_fft=config['n_fft'],
                                             win_length=config['win_length'],
                                             hop_length=config['hop_length'],
                                             n_mels=config['n_mels'])
        if config['use_logMel']:
            mel = librosa.power_to_db(mel)
            
    elif mel_type == 'simple_80mels':
        mel = librosa.feature.melspectrogram(aud,
                                             sr=sr,
                                             n_fft=config['n_fft_tron'],
                                             win_length=config['win_length_tron'],
                                             hop_length=config['hop_length_tron'],
                                             n_mels=config['n_mels_tron'])
        if config['use_logMel']:
            mel = librosa.power_to_db(mel)
            
    elif mel_type == 'transform':
        audio_norm = torch.Tensor(aud)
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        mel = STFT.mel_spectrogram(audio_norm).squeeze().data.cpu().numpy()

    return mel.astype(np.float32)


def split_audio_ixs(n_samples, rate=STEP_SIZE_EM, min_coverage=0.75):
    """
    Create audio,mel slice indices for the audio clip 

    Args:

    Returns:

    """
    assert 0 < min_coverage <= 1

    # Compute how many frames separate two partial utterances
    samples_per_frame = int((SAMPLING_RATE * WINDOW_STEP / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = int(np.round((SAMPLING_RATE / rate) / samples_per_frame))
    assert 0 < frame_step, "The rate is too high"
    assert frame_step <= H_L, "The rate is too low, it should be %f at least" % \
        (SAMPLING_RATE / (samples_per_frame * H_L))

    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - H_L + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + H_L])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))

    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / \
        (last_wav_range.stop - last_wav_range.start)
    if coverage < min_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]
    return wav_slices, mel_slices


def plot_confusion_matrix(preds, labels, label_names=None, normalize='true'):
    """
    Summary:

    Args:

    Returns:

    """
    cm = confusion_matrix(preds, labels, normalize=normalize)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    tick_marks = np.arange(len(label_names))
    cmap_min, cmap_max = plt.cm.Blues(0), plt.cm.Blues(256)

    plt.xticks(tick_marks, label_names, rotation=45)
    plt.yticks(tick_marks, label_names)
    plt.title("Ad-Detection Confusion Matrix")
    plt.tight_layout()
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if bool(normalize.capitalize()):
        fmt = '.2f'
    else:
        fmt = 'd'
    thresh = (cm.max() + cm.min()) / 2.0
    for i, j in product(range(len(label_names)), range(len(label_names))):
        color = cmap_max if cm[i, j] < thresh else cmap_min
        plt.text(i, j, format(cm[i, j], fmt),
                 horizontalalignment="center", color=color)
    plt.show()


class HDF5TorchDataset(data.Dataset):
    """
    Class for creating an Torch dataset based with random mel selection form HDF5 file


    Attributes
    ----------
    ...

    Methods
    -------
    ...

    """

    def __init__(self, data, device=torch.device('cpu')):
        hdf5_file = os.path.join(PROC_DATA_DIR, '{}.hdf5'.format(data))
        self.hdf5_file = h5py.File(hdf5_file, 'r')
        self.accents = self.hdf5_file.keys()
        with open('src/config.yaml', 'r') as f:
            self.config = safe_load(f.read())
        self.device = device

    def __len__(self):
        return len(self.accents) + int(1e4)

    def _get_acc_uttrs(self):
        while True:
            rand_accent = sample(list(self.accents), 1)[0]
            if self.hdf5_file[rand_accent].__len__() > 0:
                break
        wavs = list(self.hdf5_file[rand_accent])
        while len(wavs) < self.config['uttr_count']:
            wavs.extend(sample(wavs, 1))

        rand_wavs = sample(wavs, self.config['uttr_count'])
        rand_accent_ix = list(self.accents).index(rand_accent)
        rand_uttrs = []
        labels = []
        for wav in rand_wavs:
            wav_ = self.hdf5_file[rand_accent][wav]

            rix = randint(
                0, wav_.shape[1] - self.config['sliding_win_size'])

            ruttr = wav_[:, rix:rix +
                         self.config['sliding_win_size']]

            ruttr = torch.Tensor(ruttr)
            rand_uttrs.append(ruttr)
            labels.append(rand_accent_ix)
        return rand_uttrs, labels

    def __getitem__(self, ix=0):
        rand_uttrs, labels = self._get_acc_uttrs()
        rand_uttrs = torch.stack(rand_uttrs).to(device=self.device)
        labels = torch.LongTensor(labels).to(device=self.device)
        return rand_uttrs, labels

    def collate(self, data):
        pass


def load_audio(fname, sample_rate=None):
    """
    Load any give audio

    Args:

    Returns:

    """
    aud, sr = librosa.load(fname, sr=None)
    if sample_rate is None:
        sample_rate = config['sampling_rate']
    aud, sr = preprocess_aud(aud, sr)
    return aud


def write_hdf5(out_file, data):
    """
   Write the provieded dict to hdf5 file

    Args:

    Returns:

    """
    proc_file = h5py.File(out_file, 'w')
    print(proc_file)
    for g in data:
        group = proc_file.create_group(g)
        for datum in data[g]:
            try:
                group.create_dataset("mel_spects_{}".format(datum[0]),
                                     data=datum[1])
            except Exception as e:
                print(group.name, datum[0], e)

    proc_file.close()
    exit()


def gender_from_speaker_id_RAVDESS(config, speaker_id):
    """Obtain the gender from speaker id for RAVDESS dataset

    Parameters
    ----------
    config : dictionary 
        dictionary consisting of configurations loaded from config.yml
    speaker_id : nd-array (N samples, )
        numpy array with speaker id's between 01 and 24

    Returns
    -------
    gender_id : nd-array (N samples, )
        numpy array representing gender of the speakers, 0 for Male and 1 for Female speakers 

    """

    male_actors = [int(entry) for entry in config['male_actors']]
    female_actors = [int(entry) for entry in config['female_actors']]

    gender_id = []
    for id in speaker_id:
        if id in male_actors:
            gender_id.append(0)
        elif id in female_actors:
            gender_id.append(1)

    gender_id = np.array(gender_id, dtype=np.int).reshape(-1, )
    return gender_id


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C

def fix_length_spectrogram(specgram, mel_seg_length, append_mode='edge'):
    """Append 

    Parameters
    ----------
    specgram : nd-array spectrogram (axis0 - mels, axis1 - time)
    mel_seg_length : Length to which all the mel-spectrograms to modify 
    append_mode : if the specgram is shorter than mel_seg_length in axis1 then append using this mode. modes to select from np.pad

    Returns
    -------
    fixed length mel-spectrogram
    """
    slice_ind = abs(specgram.shape[1] - mel_seg_length) // 2
                            
    if specgram.shape[1] < mel_seg_length:
        specgram = librosa.util.fix_length(specgram, mel_seg_length, axis=1, mode=append_mode)
    else:
        specgram = specgram[:, slice_ind:slice_ind + mel_seg_length]
    
    return specgram

def return_balanced_data_indices(labels):
    """ Balance the number of classes based on the label data and return balanced indices 
    """
    rus = RandomUnderSampler()
    rus.fit_resample(labels, labels)
    return rus.sample_indices_

if __name__ == "__main__":
    
    structure()
    dataset = HDF5TorchDataset(
        config['train_dir']
    )
    loader = data.DataLoader(dataset, 4)
    for y in loader:
        print(y[0])
        print(y[1])
        break
