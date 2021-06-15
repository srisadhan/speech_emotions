import os
import sys
#nopep8
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())
import argparse
from src.resemblyzer.voice_encoder import VoiceEncoder
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import librosa
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import floor
from torch.utils.tensorboard import SummaryWriter
from yaml import safe_load

import soundfile

# from src.data.data_utils import ImbalancedDatasetSampler
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())
from src.datasets.data_utils import H_L, STEP_SIZE_EM, split_audio_ixs, mel_spectrogram, preprocess_aud, HDF5TorchDataset  # noqa
np.random.seed(42)


class SPEAKER_ENCODER(nn.Module):
    """
    Attributes
    ----------
    ...

    Methods
    -------
    ...

    """

    def __init__(self,
                 input_shape=(160, 40),
                 load_model=False,
                 epoch=0,
                 dataset_train='',
                 dataset_val='',
                 device=torch.device('cpu'),
                 loss_=None, mode='train'):
        super(SPEAKER_ENCODER, self).__init__()
        self.input_shape = input_shape
        with open('src/config.yaml', 'r') as f:
            self.config_yml = safe_load(f.read())

        model_save_dir = os.path.join(self.config_yml['model_save_dir'],
                                      'SPEAKER_ENCODER')
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        self.model_save_string = os.path.join(
            model_save_dir, self.__class__.__name__ + '_Epoch_{}.pt')

        log_dir = os.path.join(
            self.config_yml['runs_dir'],
            '{}_{}'.format(dataset_train, self.__class__.__name__))
        self.writer = SummaryWriter(log_dir=os.path.join(
            log_dir, "run_{}".format(
                len(os.listdir(log_dir)) if os.path.exists(log_dir) else 0)))

        self.device = device

        if mode == 'train':
            self.dataset_train = HDF5TorchDataset(dataset_train, device=device)
            if dataset_val == '':
                self.dataset_val = self.dataset_train
            else:
                self.dataset_val = HDF5TorchDataset(
                    args.dataset_val, device=device)

        self.lstm = nn.LSTM(self.config_yml['n_mels'],
                            self.config_yml['hidden_size'],
                            self.config_yml['num_layers'],
                            batch_first=True,
                            bidirectional=False)
        self.linear = nn.Linear(1*self.config_yml['hidden_size'],
                                self.config_yml['embedding_size'])

        self.similarity_weight = torch.nn.Parameter(
            torch.Tensor([10.0]), requires_grad=True)
        self.similarity_bias = torch.nn.Parameter(
            torch.Tensor([-5.0]), requires_grad=True)

        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.load_model = load_model
        self.epoch = epoch
        self.loss_ = loss_
        self.opt = None
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, frames):

        _, (x, _) = self.lstm(frames)  # lstm out,hidden,
        # x = x[:, -1]  #last layer -> embeds

        x = self.linear(x[-1])
        x = self.relu(x)
        # x = torch.mean(x,dim=1)
        # x = self.relu(x)

        x = x * torch.reciprocal(torch.norm(x, dim=1, keepdim=True))

        return x

    def loss_fn(self, loss_, embeds, labels):

        lang_count = int(self.config_yml['batch_size'] /
                         self.config_yml['uttr_count'])
        embeds3d = embeds.view(lang_count, self.config_yml['uttr_count'], -1)
        dcl = self.direct_classification_loss(embeds, labels)

        centroids = torch.mean(embeds3d, dim=1)

        centroids_neg = (torch.sum(embeds3d, dim=1, keepdim=True) -
                         embeds3d) / (self.config_yml['uttr_count'] - 1)
        cosim_neg = torch.cosine_similarity(embeds,
                                            centroids_neg.view_as(embeds),
                                            dim=1).view(lang_count, -1)
        centroids = centroids.repeat(
            lang_count * self.config_yml['uttr_count'], 1)

        embeds2de = embeds.unsqueeze(1).repeat_interleave(lang_count, 1).view(
            -1, self.config_yml['embedding_size'])
        cosim = torch.cosine_similarity(embeds2de, centroids)
        cosim_matrix = cosim.view(lang_count, self.config_yml['uttr_count'],
                                  -1)
        neg_ix = list(range(lang_count))

        cosim_matrix[neg_ix, :, neg_ix] = cosim_neg

        sim_matrix = (self.similarity_weight * cosim_matrix) + \
            self.similarity_bias

        sim_matrix = sim_matrix.view(self.config_yml['batch_size'], -1)
        targets = torch.range(
            0, self.config_yml['acc_count'] - 1).repeat_interleave(
                self.config_yml['uttr_count']).long().to(device=self.device)

        ce_loss = loss_(sim_matrix, targets)

        return (ce_loss, dcl), sim_matrix

    def direct_classification_loss(self, embeds, labels):
        labels = labels.reshape(-1, 1).squeeze()
        return self.ce_loss(embeds, labels)

    def train_loop(self,
                   opt,
                   lr_scheduler,
                   loss_,
                   batch_size=1,
                   gaze_pred=None,
                   cpt=0):

        train_iterator = torch.utils.data.DataLoader(self.dataset_train,
                                                     batch_size=batch_size,
                                                     shuffle=True,
                                                     drop_last=True)

        self.val_iterator = torch.utils.data.DataLoader(self.dataset_val,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        drop_last=True)

        if self.load_model:
            self.load_model_cpt(cpt=cpt)

        for epoch in range(self.epoch, 20000):
            for i, (data, labels) in enumerate(train_iterator):
                data = data.view(
                    -1,
                    self.config_yml['MEL_CHANNELS'],
                    self.config_yml['sliding_win_size'],
                ).transpose(1, 2)

                opt.zero_grad()

                embeds = self.forward(data)
                (ce_loss,
                 dcl), sim_matrix = self.loss_fn(loss_, embeds, labels)
                self.loss = ce_loss + dcl

                self.loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)

                opt.step()
                self.writer.add_scalar('ce_loss', ce_loss.data.item(), epoch)
                self.writer.add_scalar('dcl', dcl.data.item(), epoch)
                self.writer.add_scalar('Loss', self.loss.data.item(), epoch)
                # self.writer.add_scalar('ValLoss', self.val_loss(), epoch)

                self.writer.add_scalar('EER', self.eer(sim_matrix), epoch)

            if epoch % 1 == 0:
                # self.writer.add_scalar('Loss', loss.data.item(), epoch)
                # self.writer.add_scalar('Val Loss', self.val_loss(), epoch)
                # self.writer.add_scalar('EER', self.eer(sim_matrix), epoch)
                # self.writer.add_histogram('sim', sim_matrix, epoch)

                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'loss': self.loss,
                    }, self.model_save_string.format(epoch))

    def embed(self, aud, group=True):
        aud_splits, mel_splits = split_audio_ixs(len(aud))
        max_aud_length = aud_splits[-1].stop
        if max_aud_length >= len(aud):
            aud = np.pad(aud, (0, max_aud_length - len(aud)), "constant")

        mel = mel_spectrogram(aud).astype(np.float32).T
        mels = np.array([mel[s] for s in mel_splits])
        print(mels.shape)
        mels = torch.from_numpy(mels).to(self.device)
        embeds_all = []
        with torch.no_grad():
            for i in range(0, mels.shape[0], 64):
                embeds = self.forward(
                    mels[i:min(i+64, mels.shape[0]), :, :].to(self.device))
                embeds = embeds / torch.norm(embeds, dim=1, keepdim=True)
                embeds_all.append(embeds)
        embeds = torch.cat(embeds_all)

        if group:
            embeds = torch.mean(embeds, dim=0)
            embeds = embeds / torch.norm(embeds)

        embeds = embeds.cpu().data.numpy()
        # self.train()

        return embeds, (aud_splits, mel_splits)

    def accuracy(self):
        acc = 0
        ix = 0
        for i, data in enumerate(self.val_data):
            uttrs = data[0]
            embeds = self.forward(uttrs)
            # TODO
        return (acc / ix)

    def eer(self, sim_matrix=None):
        with torch.no_grad():

            targets = F.one_hot(
                torch.arange(0, self.config_yml['acc_count']),
                num_classes=self.config_yml['acc_count']).repeat_interleave(
                    self.config_yml['uttr_count'], 1).long().T

            fpr, tpr, thresholds = roc_curve(
                targets.flatten(),
                sim_matrix.detach().flatten().cpu().numpy())

            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        return eer

    def val_loss(self):
        with torch.no_grad():
            val_loss = []

            for ix, (datum, labels) in enumerate(self.val_iterator):
                datum = datum.view(
                    -1,
                    self.config_yml['MEL_CHANNELS'],
                    self.config_yml['sliding_win_size'],
                ).transpose(1, 2)
                embeds = self.forward(datum)

                (ce_loss,
                 dcl), sim_matrix = self.loss_fn(self.loss_, embeds, labels)
                loss = ce_loss + dcl

                val_loss.append(loss)

                if ix == self.config_yml['VAL_LOSS_COUNT']:
                    break

        return torch.mean(torch.stack(val_loss)).data.item()

    def load_model_cpt(self, cpt=0, opt=None, device=torch.device('cuda')):
        self.epoch = cpt

        model_pickle = torch.load(self.model_save_string.format(self.epoch),
                                  map_location=device)
        self.load_state_dict(model_pickle['model_state'])
        if opt:
            self.opt.load_state_dict(model_pickle['optimizer_state'])
        self.global_step = model_pickle['step']
        # self.loss = model_pickle['loss']
        self.loss = None
        print("Loaded Model at epoch {},with loss {}".format(
            self.epoch, self.loss))

    def infer(self, fname, cpt=None):
        aud = preprocess(fname)
        embeds = self.embed(aud, group=True)
        return embeds

    def similarity(self, embed1, embed2):
        sim = torch.cosine_similarity(
            torch.tensor(embed1).unsqueeze(0),
            torch.tensor(embed2).unsqueeze(0)).data.item()
        # sim = embed1 @ embed2
        return sim

    def sim_matrix_infer(self, fnames, cpt):
        sim_matrix = np.zeros((len(fnames), len(fnames)))
        for i, f1 in enumerate(fnames):
            for j, f2 in enumerate(fnames):
                sim_matrix[i, j] = self.similarity(self.infer(f1, cpt),
                                                   self.infer(f2, cpt))

        return sim_matrix

    def diarize(self, aud, ref_embed):
        embed = self.embed(aud, group=True)
        sim = embed@ref_embed
        return sim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",
                        help="cpu or cuda",
                        default='cuda',
                        choices=['cpu', 'cuda'])
    parser.add_argument("--dataset_train",
                        help="path to train_dataset",
                        required=False, default='timit_4_train')
    parser.add_argument("--dataset_val",
                        help="path to val_dataset",
                        required=False, default='timit_4_train')
    parser.add_argument("--mode",
                        help="train or eval",
                        required=True,
                        choices=['train', 'eval'])
    parser.add_argument(
        "--filedir",
        help="dir with fnames to run similiarity eval,atleast 2, separted by a comma",
        type=str)
    parser.add_argument("--load_model",
                        help="to load previously saved model checkpoint",
                        default=False)
    parser.add_argument(
        "--cpt",
        help="# of the save model cpt to load, only valid if valid_cpt is true"
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    loss_ = torch.nn.CrossEntropyLoss(reduction='sum')
    encoder = SPEAKER_ENCODER(dataset_train=args.dataset_train,
                              dataset_val=args.dataset_val,
                              device=device,
                              loss_=loss_,
                              load_model=args.load_model, mode=args.mode).to(device=device)
    optimizer = torch.optim.SGD(encoder.parameters(), lr=1e-2)

    encoder.opt = optimizer
    cpt = args.cpt
    if args.load_model:
        encoder.load_model_cpt(cpt=cpt, device=device)
    lr_scheduler = None
    if args.mode == 'train':
        encoder.train_loop(optimizer,
                           lr_scheduler,
                           loss_,
                           batch_size=encoder.config_yml['acc_count'],
                           cpt=cpt)
    elif args.mode == 'vis':

        fnames = [
            os.path.join(args.filedir, x)
            for x in sorted(os.listdir(args.filedir))
        ]

        sim_matrix = encoder.sim_matrix_infer(fnames, cpt)
    elif args.mode == 'eval':
        aud1, _ = preprocess_aud(os.path.join(
            encoder.config_yml['vis_dir'], 'speaker_male.wav'))
        aud2, _ = preprocess_aud(os.path.join(
            encoder.config_yml['vis_dir'], 'speaker_female.wav'))
        
        mel1 = mel_spectrogram(aud1, mel_type='simple_40mels')
        mel2 = mel_spectrogram(aud2, mel_type='simple_40mels')
        rate = 16000
        
        mel1 = librosa.db_to_power(mel1)
        mel2 = librosa.db_to_power(mel2)
        
        aud1 = librosa.feature.inverse.mel_to_audio(mel1,
                                            sr=rate,
                                            n_fft=400,
                                            hop_length=160,
                                            win_length=400)
        aud2 = librosa.feature.inverse.mel_to_audio(mel2,
                                            sr=rate,
                                            n_fft=400,
                                            hop_length=160,
                                            win_length=400)

        
        # aud1 = librosa.feature.inverse.mel_to_audio(mel1,
        #                                     sr=rate,
        #                                     n_fft=1024,
        #                                     hop_length=256,
        #                                     win_length=1024)
        # aud2 = librosa.feature.inverse.mel_to_audio(mel2,
        #                                     sr=rate,
        #                                     n_fft=1024,
        #                                     hop_length=256,
        #                                     win_length=1024)
        
        soundfile.write(os.path.join(
            encoder.config_yml['vis_dir'], 'aud1.wav'), 
                        aud1,
                        samplerate=16000)
        soundfile.write(os.path.join(
            encoder.config_yml['vis_dir'], 'aud2.wav'), 
                        aud2,
                        samplerate=16000)
        
        embed1, _ = encoder.embed(aud1, group=True)
        embed2, _ = encoder.embed(aud2, group=True)

        print(embed1@embed2)
        # print(embed1.shape,embed2.shape)
        
        encoder = VoiceEncoder()
        embed1 = encoder.embed_utterance(aud1)
        embed2 = encoder.embed_utterance(aud2)
        print(embed1@embed2)
        
        # fig, (ax1, ax2) = plt.subplots(2, 1)
        # fig.suptitle('A tale of 2 subplots')

        # ax1.plot(range(embed1.shape[1]),embed1.squeeze(), 'o-')
        # ax1.set_xlabel('S1')

        # ax2.plot(range(embed2.shape[1]),embed2.squeeze(), '.-')
        # ax2.set_xlabel('S2`')


        # plt.scatter(range(embed1.shape[1]),embed1.squeeze(),embed2.squeeze())
        # plt.savefig(os.path.join(
        #     encoder.config_yml['vis_dir'], 'plt.png'))