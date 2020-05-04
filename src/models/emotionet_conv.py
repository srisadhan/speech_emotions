import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import floor
from torch.utils.tensorboard import SummaryWriter
from yaml import safe_load
import os
import sys
import soundfile
# from src.data.data_utils import ImbalancedDatasetSampler
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

from src.datasets.data_loaders import HDF5TorchDataset
import matplotlib.pyplot as plt
import librosa
from src.datasets.create_dataset import preprocess
np.random.seed(42)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import argparse
from torch.utils.data import TensorDataset
import webrtcvad
import torchaudio
from random import randint, sample


class EMOTIONET(nn.Module):
    def __init__(self,
                 input_shape=(64, 80),
                 load_model=False,
                 epoch=0,
                 dataset_train='gmu_4_train',
                 dataset_val='gmu_4_val',
                 device=torch.device('cpu'),
                 loss_=None,
                 beta=1.0):
        super(EMOTIONET, self).__init__()
        self.input_shape = input_shape
        with open('src/config.yaml', 'r') as f:
            self.config = safe_load(f.read())

        model_save_dir = os.path.join(self.config['model_save_dir'],
                                      dataset_train)
                
        os.makedirs(model_save_dir,exist_ok=True)
        os.makedirs(self.config['vis_dir'],exist_ok=True)

        self.model_save_string = os.path.join(
            model_save_dir, self.__class__.__name__ + '_Epoch_{}.pt')

        log_dir = os.path.join(
            self.config['runs_dir'], '{}_{}'.format(dataset_train,
                                                    self.__class__.__name__))
        self.writer = SummaryWriter(log_dir=os.path.join(
            log_dir, "run_{}".format(
                len(os.listdir(log_dir)) if os.path.exists(log_dir) else 0)))

        self.device = device

        self.dataset_train = HDF5TorchDataset(dataset_train, device=device)
        self.dataset_val = HDF5TorchDataset(dataset_val, device=device)

        self.encoder = nn.Sequential(
            nn.Conv1d(self.config['n_mels'], 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv1d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv1d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv1d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv1d(64, 256, 4, 1),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose1d(256, 64, 4),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose1d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose1d(32, self.config['n_mels'], 4, 2, 1),
        )

        self.conv_weight_init()  #call before more nn defintions

        self.mu = nn.Linear(256, self.config['emo_z_hidden_dim'])
        self.logvar = nn.Linear(256, self.config['emo_z_hidden_dim'])
        self.linear = nn.Linear(16, 256)

        torch.nn.init.zeros_(self.mu.weight)
        torch.nn.init.zeros_(self.logvar.weight)
        torch.nn.init.zeros_(self.linear.weight)

        self.load_model = load_model
        self.epoch = epoch
        self.opt = None
        self.config = dict(self.config)

        self.mce_loss = nn.MSELoss(reduction='mean')
        self.beta = beta

    def conv_weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0)

    def encode(self, x):

        x = self.encoder(x)

        x = x.view(-1, 256)

        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.empty(std.shape).normal_(0.0, 1.0).to(device=self.device)
        # eps = -20
        z = mu + (eps * std)

        return z

    def decode(self, z):

        x = self.linear(z).unsqueeze(-1)
        x = self.decoder(x)

        return x

    def forward(self, frames):
        mu, logvar = self.encode(frames)
        z = self.reparameterize(mu, logvar)

        x = self.decode(z)

        return x, mu, logvar

    def loss_fn(self, loss_, outs, mu, logvar, frames):
        ####vae loss KLD
        loss1 = -0.5 * (1 + logvar - torch.pow(mu, 2) - torch.exp(logvar))

        loss1 = torch.sum(loss1, 1).mean()
        loss1 = self.beta * loss1

        ###recon loss
        loss2 = self.mce_loss(frames, outs)

        return loss1, loss2

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
            self.epoch = epoch
            for i, data in enumerate(train_iterator):
                # data = data.view(
                #     -1,
                #     self.config['mel_seg_length'],
                #     self.config['n_mels'],
                # )

                opt.zero_grad()

                outs, mu, logvar = self.forward(data)
                loss1, loss2 = self.loss_fn(loss_, outs, mu, logvar, data)
                self.loss = loss1 + loss2

                self.loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                opt.step()
                self.writer.add_scalar('Recon Loss', loss2.data.item(), epoch)
                self.writer.add_scalar('KL div', loss1.data.item(), epoch)

            if epoch % 1000 == 0:
                aud = self.griffin_lim(outs[-1].data.numpy())
                torch.save(
                    {
                        'epoch': self.epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'loss': self.loss,
                    }, self.model_save_string.format(epoch))

    def griffin_lim_aud(self, spec):
        y = librosa.feature.inverse.mel_to_audio(
            spec,
            sr=self.config['resampling_rate'],
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length'],
            win_length=self.config['win_length'])

        soundfile.write(os.path.join(self.config['vis_dir'],
                                     '{}.wav'.format(self.epoch)),
                        y,
                        samplerate=self.config['resampling_rate'])
        return y

    def load_model_cpt(self, cpt=0, opt=None, device=torch.device('cuda')):
        self.epoch = cpt

        model_pickle = torch.load(self.model_save_string.format(self.epoch),
                                  map_location=device)
        self.load_state_dict(model_pickle['model_state_dict'])
        self.opt.load_state_dict(model_pickle['optimizer_state_dict'])
        self.epoch = model_pickle['epoch']
        self.loss = model_pickle['loss']
        print("Loaded Model at epoch {},with loss {}".format(
            self.epoch, self.loss))

    def latent_sampling(self, visualize=True):
        with torch.no_grad():
            self.eval()
            mu = torch.rand(1, 16)
            logvar = torch.rand(1, 16)
            z = self.reparameterize(mu, logvar)
            x = self.decode(z)
            self.epoch = 'sample'
            if visualize:
                S_dB = librosa.power_to_db(x[0].data.numpy(), ref=np.max)
                librosa.display.specshow(S_dB,
                                         x_axis='s',
                                         y_axis='mel',
                                         sr=16000,
                                         fmax=8000,
                                         cmap='viridis')
                plt.colorbar(format='%+2.0f dB')
                plt.title('Mel-frequency spectrogram')
                plt.tight_layout()
                plt.show()
            aud = self.griffin_lim_aud(x[0].data.numpy())
        self.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",
                        help="cpu or cuda",
                        default='cuda',
                        choices=['cpu', 'cuda'])
    parser.add_argument("--dataset_train",
                        help="path to train_dataset",
                        required=True)
    parser.add_argument("--dataset_val",
                        help="path to val_dataset",
                        required=True)
    parser.add_argument("--mode",
                        help="train or eval",
                        required=True,
                        choices=['train', 'eval', 'sample'])
    parser.add_argument("--beta",
                        help="beta factor value for kl, should be > 1",
                        default=1.0,
                        type=float)
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
    encoder = EMOTIONET(dataset_train=args.dataset_train,
                        dataset_val=args.dataset_val,
                        device=device,
                        loss_=loss_,
                        load_model=args.load_model,
                        beta=args.beta).to(device=device)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    encoder.opt = optimizer
    cpt = args.cpt
    if args.load_model:
        encoder.load_model_cpt(cpt=cpt, device=device)

    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lr_lambda=lambda x: x*0.95)

    lr_scheduler = None
    if args.mode == 'train':
        encoder.train_loop(optimizer,
                           lr_scheduler,
                           loss_,
                           batch_size=encoder.config['batch_size'],
                           cpt=cpt)
    elif args.mode == 'sample':

        encoder.latent_sampling()