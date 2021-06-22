import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # nopep8
sys.path.append(os.getcwd())  # nopep8

from yaml import safe_load
import deepdish as dd
from src.models.synthesizer.hparams import hparams
from src.models.synthesizer.utils.text import text_to_sequence
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import torch

from torch.utils.data.dataloader import DataLoader


class SynthesizerDataset(Dataset):
    def __init__(self, meldata_fpath: Path, hparams):
        print("Using inputs from:\n\t%s" % (meldata_fpath))

        self.data = dd.io.load(meldata_fpath)
        self.data_fields = list(self.data.keys())
        print(self.data_fields)

        # mel_fnames = [x[1] for x in metadata if int(x[4])]
        # mel_fpaths = [mel_dir.joinpath(fname) for fname in mel_fnames]
        # embed_fnames = [x[2] for x in metadata if int(x[4])]
        # embed_fpaths = [embed_dir.joinpath(fname) for fname in embed_fnames]
        # self.samples_fpaths = list(zip(mel_fpaths, embed_fpaths))
        # self.samples_texts = [x[5].strip() for x in metadata if int(x[4])]
        # self.metadata = metadata
        self.hparams = hparams

    def __getitem__(self, index):
        # Sometimes index may be a list of 2 (not sure why this happens)
        # If that is the case, return a single item corresponding to first element in index

        if index is list:
            index = index[0]

        embed = self.data['speaker_embeds'][index]
        mel = self.data['features'][index]

        # TODO add mel, text, speaker embeds, emotion embeds to the fields into the h5 data
        # text = self.data['text'][index]

        text = ""

        # Get the text and clean it
        text = text_to_sequence(text, self.hparams.tts_cleaner_names)

        # Convert the list returned by text_to_sequence to a numpy array
        text = np.asarray(text).astype(np.int32)

        return text, mel.astype(np.float32), embed.astype(np.float32), index

    def __len__(self):
        return len(self.data[self.data_fields[0]])


def collate_synthesizer(batch, r, hparams):
    # Text
    x_lens = [len(x[0]) for x in batch]
    max_x_len = max(x_lens)

    chars = [pad1d(x[0], max_x_len) for x in batch]
    chars = np.stack(chars)

    # Mel spectrogram
    spec_lens = [x[1].shape[-1] for x in batch]
    max_spec_len = max(spec_lens) + 1
    if max_spec_len % r != 0:
        max_spec_len += r - max_spec_len % r

    # WaveRNN mel spectrograms are normalized to [0, 1] so zero padding adds silence
    # By default, SV2TTS uses symmetric mels, where -1*max_abs_value is silence.
    if hparams.symmetric_mels:
        mel_pad_value = -1 * hparams.max_abs_value
    else:
        mel_pad_value = 0

    mel = [pad2d(x[1], max_spec_len, pad_value=mel_pad_value) for x in batch]
    mel = np.stack(mel)

    # Speaker embedding (SV2TTS)
    embeds = [x[2] for x in batch]

    # Index (for vocoder preprocessing)
    indices = [x[3] for x in batch]

    # Convert all to tensor
    chars = torch.tensor(chars).long()
    mel = torch.tensor(mel)
    embeds = torch.tensor(embeds)

    return chars, mel, embeds, indices


def pad1d(x, max_len, pad_value=0):
    return np.pad(x, (0, max_len - len(x)), mode="constant", constant_values=pad_value)


def pad2d(x, max_len, pad_value=0):
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode="constant", constant_values=pad_value)


if __name__ == "__main__":
    with open('src/config.yaml', 'r') as f:
        config = safe_load(f.read())

    print(config['const_40mel_simple'])
    syn_dataset = SynthesizerDataset(config['const_40mel_simple'], hparams)
    r = 12
    syn_iterator = DataLoader(
        syn_dataset, collate_fn=lambda batch: collate_synthesizer(batch, r, hparams))
    for x in syn_iterator:
        print(x)
        break
