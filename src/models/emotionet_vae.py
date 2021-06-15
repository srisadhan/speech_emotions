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

from src.datasets.data_loaders import EmotionDataset
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

# Variational Autoencoder that takes a partial segment of emotional speech spectrogram as an input - 80x64 and 16 latent space embeddings (mu, sigma)
class EmotionNet_VAE(nn.Module):
    def __init__(self,
                 input_shape=(64, 80),
                 load_model=False,
                 epoch=0,
                 dataset_train='gmu_4_train',
                 dataset_val='gmu_4_val',
                 device=torch.device('cpu'),
                 loss_=None,
                 beta=1.0):
        super(EmotionNet_VAE, self).__init__()
        self.input_shape = input_shape
        with open('src/config.yaml', 'r') as f:
            self.config = safe_load(f.read())

        model_log_dir = os.path.join(
            self.config['model_save_dir'], '{}_{}'.format(dataset_train,
                                                    self.__class__.__name__))
        run_log_dir = os.path.join(
            self.config['runs_dir'], '{}_{}'.format(dataset_train,
                                                    self.__class__.__name__))
        
        if not load_model:  
            model_save_dir = os.path.join(os.path.join(
                                      model_log_dir, "run_{}".format(
                                      len(os.listdir(model_log_dir)) if os.path.exists(model_log_dir) else 0))
                                      )
            self.model_save_string = os.path.join(
                    model_save_dir, self.__class__.__name__ + '_Epoch_{}.pt')
            
            os.makedirs(model_save_dir,exist_ok=True)
            os.makedirs(self.config['vis_dir'],exist_ok=True)
            
            self.writer = SummaryWriter(log_dir=os.path.join(
                                        run_log_dir, "run_{}".format(
                                        len(os.listdir(run_log_dir)) if os.path.exists(run_log_dir) else 0)))
        else:
            model_save_dir = os.path.join(os.path.join(
                                      model_log_dir, "run_{}".format(
                                      len(os.listdir(model_log_dir)) - 1 if os.path.exists(model_log_dir) else 0))
                                      )
            self.model_save_string = os.path.join(
                model_save_dir, self.__class__.__name__ + '_Epoch_{}.pt')

        self.device = device

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

        #FIXME: weight initilization changed
        # torch.nn.init.kaiming_normal_(self.logvar.weight)
        # torch.nn.init.kaiming_normal_(self.linear.weight)
        
        self.load_model = load_model
        self.epoch = epoch
        self.opt = None
        self.config = dict(self.config)

        # self.mse_loss = nn.MSELoss(reduction='mean')
        #FIXME: BCELoss requires the target to be in the range (0,1),
        #FIXME: Can CTCLoss be used (loss between temporal time series data)
        self.mse_loss = nn.CrossEntropyLoss(reduction='mean')
        
        self.beta = beta

    def conv_weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, nn.Conv1d):
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
        mu  = mu.to(device=self.device)
        std = torch.exp(logvar).to(device=self.device)
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
        loss2 = self.mse_loss(outs, frames)
        # loss2 = binary_cross_entropy(outs, frames)

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

        for epoch in range(self.epoch, self.config['train_epochs']):
            self.epoch = epoch
            for i, data in enumerate(train_iterator):

                opt.zero_grad()

                outs, mu, logvar = self.forward(data)
                loss1, loss2 = self.loss_fn(loss_, outs, mu, logvar, data)
                self.loss = loss1 + loss2
                
                self.loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                opt.step()
                
                self.writer.add_scalar('Recon Loss', loss2.data.item(), epoch)
                self.writer.add_scalar('KL div', loss1.data.item(), epoch)
                
            if epoch % 100 == 0:
                print("Device: {}, Epoch: {}, KL: {}, ReconsLoss: {}".format(self.device, epoch, loss1, loss2))
                aud = self.griffin_lim_aud(outs[-1].cpu().data.numpy())
                torch.save(
                    {
                        'epoch': self.epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'loss': self.loss,
                    }, self.model_save_string.format(epoch))

    def griffin_lim_aud(self, spec):
        if self.config['use_logMel']:
            spec = librosa.db_to_power(spec)
        else:
            spec = spec

        y = librosa.feature.inverse.mel_to_audio(
            spec,
            sr=self.config['resampled_rate'],
            n_fft=self.config['n_fft'],
            hop_length=self.config['hop_length'],
            win_length=self.config['win_length'])

        soundfile.write(os.path.join(self.config['vis_dir'],
                                     '{}.wav'.format(self.epoch)),
                        y,
                        samplerate=self.config['resampled_rate'])
        return y

    def load_model_cpt(self, cpt=0, opt=None, device=torch.device('cuda:0')):
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
            
            #FIXME: only one variable should be sampled insted of all
            # mu = torch.zeros(1, 16)
            # logvar = torch.zeros(1, 16)
            
            # mu[0,6] = torch.rand(1)
            # logvar[0,6] = torch.rand(1)
            print(mu, logvar)
            
            z = self.reparameterize(mu, logvar)
            x = self.decode(z)
            self.epoch = 'sample'
            if visualize:
                if self.config['use_logMel']:
                    S_dB = x[0].cpu().data.numpy()
                else:
                    S_dB = librosa.power_to_db(x[0].cpu().data_numpy())
                    
                # S_dB = librosa.power_to_db(x[0].cpu().data.numpy(), ref=np.max)
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
            aud = self.griffin_lim_aud(x[0].cpu().data.numpy())
        self.train()

# Variational Autoencoder that takes complete sequence as an input - 80x128 and 16 latent space embeddings (mu, sigma)
class EmotionNet_VAE2(nn.Module):
    def __init__(self,
                #  input_shape=(64, 80),
                 input_shape=(248, 80),
                 load_model=False,
                 epoch=0,
                 dataset_train='gmu_4_train',
                 dataset_val='gmu_4_val',
                 device=torch.device('cpu'),
                 loss_=None,
                 beta=1.0):
        super(EmotionNet_VAE2, self).__init__()
        self.input_shape = input_shape
        with open('src/config.yaml', 'r') as f:
            self.config = safe_load(f.read())

        
        model_log_dir = os.path.join(
            self.config['model_save_dir'], '{}_{}'.format(dataset_train,
                                                    self.__class__.__name__))
        run_log_dir = os.path.join(
            self.config['runs_dir'], '{}_{}'.format(dataset_train,
                                                    self.__class__.__name__))
        
        if not load_model:  
            model_save_dir = os.path.join(os.path.join(
                                      model_log_dir, "run_{}".format(
                                      len(os.listdir(model_log_dir)) if os.path.exists(model_log_dir) else 0))
                                      )
            self.model_save_string = os.path.join(
                    model_save_dir, self.__class__.__name__ + '_Epoch_{}.pt')
            
            os.makedirs(model_save_dir,exist_ok=True)
            os.makedirs(self.config['vis_dir'],exist_ok=True)
            
            self.writer = SummaryWriter(log_dir=os.path.join(
                                        run_log_dir, "run_{}".format(
                                        len(os.listdir(run_log_dir)) if os.path.exists(run_log_dir) else 0)))
        else:
            model_save_dir = os.path.join(os.path.join(
                                      model_log_dir, "run_{}".format(
                                      len(os.listdir(model_log_dir)) - 1 if os.path.exists(model_log_dir) else 0))
                                      )
            self.model_save_string = os.path.join(
                model_save_dir, self.__class__.__name__ + '_Epoch_{}.pt')       

        self.device = device
        
        self.encoder = nn.Sequential(nn.Conv1d(self.config['n_mels'], 64, 4, 2, 1, 1), 
                                     nn.ReLU(True),                           
                                     nn.Conv1d(64, 32, 4, 2, 1), 
                                    #  nn.ReLU(True), 
                                    #  nn.Conv1d(62, 32, 4, 2, 1), 
                                     nn.ReLU(True), 
                                     nn.Conv1d(32, 16, 4, 2, 1), 
                                     nn.ReLU(True), 
                                     nn.Conv1d(16, 16, 4, 2, 1), 
                                     nn.ReLU(True),                     
                                     nn.Conv1d(16, 128, 4, 2, 1), 
                                     nn.ReLU(True),
                                     nn.Conv1d(128, 256, 3, 2), 
                                     nn.ReLU(True)
                                     )

        self.decoder = nn.Sequential(nn.ReLU(True), nn.ConvTranspose1d(256, 128, 3),
                                    nn.ReLU(True), nn.ConvTranspose1d(128, 16, 4, 2),
                                    nn.ReLU(True), nn.ConvTranspose1d(16, 16, 4, 2, 1),
                                    nn.ReLU(True), nn.ConvTranspose1d(16, 32, 4, 2, 1),
                                    nn.ReLU(True), nn.ConvTranspose1d(32, 64, 4, 2, 1),
                                    # nn.ReLU(True), nn.ConvTranspose1d(64, 124, 4, 2, 1),
                                    nn.ReLU(True), nn.ConvTranspose1d(64, self.config['n_mels'], 4, 2, 1),
                                    )
        
        self.conv_weight_init()  #call before more nn defintions

        self.inter_linear = nn.Linear(256, self.config['emo_z_hidden_dim'])
        self.invert_linear = nn.Linear(self.config['latent_vars'], 256)

        # torch.nn.init.zeros_(self.mu.weight)
        # torch.nn.init.zeros_(self.logvar.weight)
        # torch.nn.init.zeros_(self.linear.weight)

        #FIXME: weight initilization changed
        torch.nn.init.kaiming_normal_(self.inter_linear.weight)
        torch.nn.init.kaiming_normal_(self.invert_linear.weight)
        
        self.load_model = load_model
        self.epoch = epoch
        self.opt = None
        self.config = dict(self.config)

        self.mse_loss = nn.MSELoss(reduction='mean')
        #FIXME: BCELoss requires the target to be in the range (0,1),
        #FIXME: Can CTCLoss be used (loss between temporal time series data)
        # self.mse_loss = nn.CrossEntropyLoss(reduction='sum')
        
        self.beta = beta

    def conv_weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, nn.Conv1d):
                    torch.nn.init.kaiming_normal(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0)

    def encode(self, x):

        x = self.encoder(x)
        x = self.inter_linear(x.view(-1, 256))
        
        mu     = x[:, :8]
        logvar = x[:, 8:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        mu  = mu.to(device=self.device)
        std = torch.exp(torch.div(logvar, 2)).to(device=self.device)
        eps = torch.empty(std.shape).normal_(0.0, 1.0).to(device=self.device)
        
        z = mu + (eps * std)

        return z

    def decode(self, z):

        x = self.invert_linear(z).unsqueeze(-1) # similar to view(-1, 256, 1)
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

        loss1 = torch.sum(loss1, 1).mean(0, True)
        loss1 = self.beta * loss1

        ###recon loss
        loss2 = self.mse_loss(outs, frames)
        # loss2 = binary_cross_entropy(outs, frames)

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


        if self.load_model:
            self.load_model_cpt(cpt=cpt)

        for epoch in range(self.epoch, self.config['train_epochs']):
            self.epoch = epoch
            for i, data in enumerate(train_iterator):
                
                opt.zero_grad()
                
                outs, mu, logvar = self.forward(data)
                loss1, loss2 = self.loss_fn(loss_, outs, mu, logvar, data)
                self.loss = loss1 + loss2
                
                self.loss.backward()
        
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                opt.step()
                
                self.writer.add_scalar('Recon Loss', loss2.data.item(), epoch)
                self.writer.add_scalar('KL div', loss1.data.item(), epoch)
                
            if epoch % 100 == 0:
                print("Device: {}, Epoch: {}, KL: {}, ReconsLoss: {}".format(self.device, epoch, loss1, loss2))
                aud = self.griffin_lim_aud(outs[-1].cpu().data.numpy())
                torch.save(
                    {
                        'epoch': self.epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'loss': self.loss,
                    }, self.model_save_string.format(epoch))

    def griffin_lim_aud(self, spec):
        if self.config['use_logMel']:
            spec = librosa.db_to_power(spec)
        else:
            spec = spec
            
        y = librosa.feature.inverse.mel_to_audio(spec,
                                            sr=self.config['resampled_rate'],
                                            n_fft=self.config['n_fft'],
                                            hop_length=self.config['hop_length'],
                                            win_length=self.config['win_length'])

        soundfile.write(os.path.join(self.config['vis_dir'],
                                    '{}.wav'.format(self.epoch)),
                                    y,
                                    samplerate=self.config['resampled_rate'])
        return y

    def load_model_cpt(self, cpt=0, opt=None, device=torch.device('cuda:0')):
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
            
            for i in range(len(self.config['emotions'])):
                mu = torch.zeros(1, 8)
                logvar = torch.zeros(1, 8)
                
                mu[0,i] = 0.5 #torch.rand(1)
                logvar[0,i] = 0.5 #torch.rand(1)
                print(mu, logvar)
                
                z = self.reparameterize(mu, logvar)
                x = self.decode(z)
                self.epoch = 'sample_emotion_' + str(i + 1) 
                if visualize:
                    if self.config['use_logMel']:
                        S_dB = x[0].cpu().data.numpy()
                    else:
                        S_dB = librosa.power_to_db(x[0].cpu().data.numpy())
                    
                    plt.figure()
                    # S_dB = librosa.power_to_db(x[0].cpu().data.numpy(), ref=np.max)
                    librosa.display.specshow(S_dB, #x[0].cpu().data.numpy(),
                                            x_axis='s',
                                            y_axis='mel',
                                            sr=16000,
                                            fmax=8000,
                                            cmap='viridis')
                    plt.colorbar(format='%+2.0f dB')
                    plt.title('Mel-frequency spectrogram')
                    plt.tight_layout()  
                aud = self.griffin_lim_aud(x[0].cpu().data.numpy())
                
        plt.show()
        self.train()
    
    def latent_testing(self, visualize=True):
        with torch.no_grad():
            self.eval()
            
            for i in range(len(self.config['emotions'])):
                # z = torch.normal(0.0, 1.0, size=(1, self.config['latent_vars'])).to(device=self.device)
                z = torch.zeros((1, self.config['latent_vars']))
                z[0, i] = torch.normal(0.0, 3.0, size=(1,1))
                
                z = z.to(device=self.device)
                x = self.decode(z)
                self.epoch = 'sample_emotion_' + str(i + 1) 
                if visualize:
                    if self.config['use_logMel']:
                        S_dB = x[0].cpu().data.numpy()
                    else:
                        S_dB = librosa.power_to_db(x[0].cpu().data.numpy())
                    
                    plt.figure()
                    # S_dB = librosa.power_to_db(x[0].cpu().data.numpy(), ref=np.max)
                    librosa.display.specshow(S_dB, #x[0].cpu().data.numpy(),
                                            x_axis='s',
                                            y_axis='mel',
                                            sr=16000,
                                            fmax=8000,
                                            cmap='viridis')
                    # plt.colorbar(format='%+2.0f dB')
                    # plt.title('Mel-frequency spectrogram')
                    plt.xticks([])
                    plt.yticks([])
                    plt.xlabel('')
                    plt.ylabel('')
                    plt.tight_layout()  
                aud = self.griffin_lim_aud(x[0].cpu().data.numpy())
                
        plt.show()
        

def binary_cross_entropy(r, x):
    "Drop in replacement until PyTorch adds `reduce` keyword."
    return -torch.sum(x * torch.log(r + 1e-8) + (1 - x) * torch.log(1 - r + 1e-8), dim=-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--device",
    #                     help="cpu or cuda",
    #                     default='cuda',
    #                     choices=['cpu', 'cuda'])
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device(args.device)
    loss_ = torch.nn.CrossEntropyLoss(reduction='sum')
    encoder = EmotionNet_VAE(dataset_train=args.dataset_train,
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