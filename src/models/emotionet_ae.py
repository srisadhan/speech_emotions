from numpy.lib.npyio import save
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReflectionPad1d
from torch.utils.tensorboard import SummaryWriter
from yaml import safe_load
import os
import sys
import soundfile
# from src.data.data_utils import ImbalancedDatasetSampler
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import librosa
np.random.seed(42)
import deepdish as dd
from pathlib import Path
import yaml
import umap 

# Autoencoder that takes 40x128 as input and has 256 latent space embeddings 
class EmotionNet_AE(nn.Module):
    
    def __init__(self,
                 n_mels,
                 load_model=False,
                 epoch=1,
                 device=torch.device('cpu'),
                 use_speaker_embeds=False):
        
        super(EmotionNet_AE, self).__init__()
        with open('src/config.yaml', 'r') as f:
            self.config = safe_load(f.read())

        self.device = device
        self.n_mels = n_mels
        self.use_speaker_embeds = use_speaker_embeds
        
        self.encoder = nn.Sequential(self.conv_block(self.n_mels, 32, 32, 1, 0),
                                    # nn.MaxPool1d(kernel_size=8, stride=2),
                                    self.conv_block(32, 64, 16, 1, 0),
                                    # # nn.MaxPool1d(kernel_size=8, stride=2),
                                    self.conv_block(64, 128, 8, 1, 0),
                                    # # nn.MaxPool1d(kernel_size=8, stride=2),
                                    self.conv_block(128, self.config['embedding_size'], 8, 1, 0),
                                    nn.MaxPool1d(kernel_size=128)
                                    )
        
        self.decoder = nn.Sequential(self.conv_transpose(self.config['embedding_size'], 128, 8, 2, 0),
                                    nn.LeakyReLU(),
                                    # nn.MaxUnpool1d(kernel_size=8, stride=2),
                                    self.conv_transpose(128, 64, 8, 1, 0),
                                    nn.LeakyReLU(),
                                    # nn.MaxUnpool1d(kernel_size=8, stride=2),
                                    self.conv_transpose(64, 32, 16, 1, 0),
                                    nn.LeakyReLU(),
                                    # nn.MaxUnpool1d(kernel_size=8, stride=2),
                                    self.conv_transpose(32, self.n_mels, 32, 1, 0)
                                    )
        
        
        self.weight_init()  #call before more nn definitions

        if use_speaker_embeds:
            self.linear1 = nn.Linear(self.config['embedding_size'] * 2, self.config['n_emotions'])
        else:    
            self.linear1 = nn.Linear(self.config['embedding_size'], self.config['n_emotions'])
       
        self.load_model = load_model
        self.epoch = epoch
        self.optimizer = None
        self.config = dict(self.config)

        self.alpha = 10
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        
        if self.n_mels == 80:
            # load waveglow
            self.waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
            self.waveglow = self.waveglow.remove_weightnorm(self.waveglow)
            self.waveglow.to(self.device)
            self.waveglow.eval()
        
    def conv_block(self, in_channels, out_channels, kernel_size, stride, dropout_prob):
        # pad the layers such that the output has the same size of input 
        if (kernel_size - 1) % 2 == 0:
            pad_left  = int((kernel_size - 1) / 2)
            pad_right = int((kernel_size - 1) / 2)
        else:
            pad_left  = int(kernel_size / 2 )
            pad_right = int(kernel_size / 2 - 1)

        conv = nn.Sequential(
            nn.ConstantPad1d(padding=(pad_left, pad_right), value=0),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.LeakyReLU(),
            # nn.Dropout(p=dropout_prob),
        )

        return conv
    
    def conv_transpose(self, in_channels, out_channels, kernel_size, stride, dropout_prob):
        # pad the layers such that the output has the same size of input 
        if (kernel_size - 1) % 2 == 0:
            pad_left  = int((kernel_size - 1) / 2)
            pad_right = int((kernel_size - 1) / 2)
        else:
            pad_left  = int(kernel_size / 2 )
            pad_right = int(kernel_size / 2 - 1)

        conv = nn.Sequential(
            nn.ConstantPad1d(padding=(pad_left, pad_right), value=0),
            nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            # nn.Dropout(p=dropout_prob)
        )

        return conv
    
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                    torch.nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0)

    def forward(self, x, speaker_embeds):
        # AutoEncoder
        embeds  = self.encoder(x)
        recon_x = self.decoder(embeds)

        embeds = embeds.view(-1, self.config['embedding_size'])
        
        embeds = embeds.clone() / (torch.norm(embeds, dim=1, keepdim=True) + 1e-5)
        
        if self.use_speaker_embeds:
            embeds = torch.cat((embeds, speaker_embeds.squeeze()), dim=1)# concatenate the AE embeddings with the speaker embedding
        # A linear layer with ReLu activation to classifiy the emotions
        output = F.relu(self.linear1(embeds))
        
        return embeds, recon_x, output

    def reconstruction_loss(self, x, recon_x):
        return self.mse_loss(recon_x, x)

    def direct_classification_loss(self, output, labels):
        labels = labels
        return self.ce_loss(output, labels)
    
    def loss_fn(self, x, recon_x, output, labels):
        # reconstruction loss from the waveforms
        recon_loss = self.reconstruction_loss(x, recon_x)
        # cross entropy emotion prediction loss
        pred_loss = self.direct_classification_loss(output, labels) 
       
        return (recon_loss + self.alpha*pred_loss)
    
    def accuracy(self, output, labels):
        predictions = torch.argmax(output, dim=1)   
        correct = torch.sum(torch.eq(predictions, labels))

        accuracy = 100 * correct / labels.shape[0]
        
        return accuracy.detach().cpu().numpy()
    
    def train_model(self,
                   train_dataloader,
                   optimizer,
                   device,
                   lr_scheduler=None,
                   load_model=False,
                   checkpoint=None):

        self.device = device

        model_log_dir = os.path.join(
            self.config['model_save_dir'], '{}'.format(self.__class__.__name__))
        run_log_dir = os.path.join(
            self.config['runs_dir'], '{}'.format(self.__class__.__name__))
        
        if not load_model:  
            model_save_dir = os.path.join(os.path.join(
                                      model_log_dir, "mel_{}_run_{}".format(self.n_mels,
                                      len(os.listdir(model_log_dir)) if os.path.exists(model_log_dir) else 0))
                                      )
            self.model_save_string = os.path.join(
                    model_save_dir, self.__class__.__name__ + '_Epoch_{}.pt')
            
            os.makedirs(model_save_dir, exist_ok=True)
            os.makedirs(self.config['vis_dir'], exist_ok=True)
            
            self.writer = SummaryWriter(log_dir=os.path.join(
                                        run_log_dir, "mel_{}_run_{}".format(self.n_mels,
                                        len(os.listdir(run_log_dir)) if os.path.exists(run_log_dir) else 0)))
        else:
            model_save_dir = os.path.join(os.path.join(
                                      model_log_dir, "mel_{}_run_{}".format(self.n_mels,
                                      len(os.listdir(model_log_dir)) - 1 if os.path.exists(model_log_dir) else 0))
                                      )
            self.model_save_string = os.path.join(
                model_save_dir, self.__class__.__name__ + '_Epoch_{}.pt')       


        if self.load_model:
            self.load_model_from_dict(checkpoint=checkpoint)

        for epoch in range(self.epoch, self.config['train_epochs']):
            self.epoch = epoch
            
            # set the model to training mode
            self.train(mode=True)
            loss_vec, accuracy_vec = [], []
            embeds_vec, label_vec = [], []
            for i, data in enumerate(train_dataloader):
                
                optimizer.zero_grad()
                
                embeds, recon_x, output = self.forward(data['features'].to(self.device), data['speaker_embeds'].to(self.device))
                self.loss= self.loss_fn(data['features'].to(self.device), recon_x, output, data['labels'].to(self.device))
                                
                self.loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                optimizer.step()
                
                accuracy = self.accuracy(output, data['labels'].to(self.device))
                
                loss_vec.append(self.loss.data.item())
                accuracy_vec.append(accuracy)
                embeds_vec.append(embeds.detach().cpu().numpy())
                label_vec.append(data['labels'].cpu().numpy())
                
            self.writer.add_scalar('Running loss', np.mean(loss_vec), epoch)
            self.writer.add_scalar('Accuracy', np.mean(accuracy_vec), epoch)
                
            if lr_scheduler:
                if epoch % 49 == 0:
                    lr_scheduler.step()
                
            if epoch % 500 == 0:
                print("Device: {}, Epoch: {}, Loss: {}, Accuracy: {}".format(self.device, epoch, np.mean(loss_vec), np.mean(accuracy_vec)))
                
                if self.n_mels == 40:
                    aud = self.griffin_lim_aud(recon_x[-1].cpu().data.numpy(), save_audio=True)
                else:
                    aud = self.waveglow_aud(recon_x[-1].unsqueeze(dim=0), save_audio=True)
                
                torch.save(
                    {
                        'epoch': self.epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': self.loss,
                        'use_speaker_embeds': self.use_speaker_embeds,
                    }, self.model_save_string.format(epoch))
                
                # reducer = umap.UMAP()
                # umap_embeds = reducer.fit_transform(np.concatenate(embeds_vec, axis=0))
                
                # plt.figure()
                # plt.scatter(umap_embeds[:, 0], umap_embeds[:, 1], c=np.concatenate(label_vec, axis=0), cmap='viridis')
                # plt.pause(.1)
                

    def griffin_lim_aud(self, spec, save_audio=False):
        if self.config['use_logMel']:
            spec = librosa.db_to_power(spec)
        else:
            spec = spec
            
        y = librosa.feature.inverse.mel_to_audio(spec,
                                            sr=self.config['resampled_rate'],
                                            n_fft=self.config['n_fft'],
                                            hop_length=self.config['hop_length'],
                                            win_length=self.config['win_length'])

        if save_audio:
            savepath = os.path.join(self.config['vis_dir'], 'Mel_{}'.format(str(self.n_mels)))
            os.makedirs(savepath, exist_ok=True)
            
            savepath = os.path.join(savepath, 'epoch_{}.wav'.format(self.epoch))
            
            soundfile.write(savepath, y, samplerate=self.config['resampled_rate'])
        return y

    def waveglow_aud(self, spec, save_audio=False):
        """ Convert the 80 mel spectrogram to audio using NVIDIA's Waveglow """
        with torch.no_grad():
            audio = self.waveglow.infer(spec)
        audio = audio[0].data.cpu().numpy()
        
        if save_audio:
            savepath = os.path.join(self.config['vis_dir'], 'Mel_{}'.format(str(self.n_mels)))
            os.makedirs(savepath, exist_ok=True)
            
            savepath = os.path.join(savepath, 'epoch_{}.wav'.format(self.epoch))
            soundfile.write(savepath, audio, samplerate=22050)
        
        return audio
        
    def load_model_from_dict(self, checkpoint):
        """ Load the model from the checkpoint by filtering out the unnecessary parameters"""
        model_dict = self.state_dict()
        # filter out unnecessary keys in the imported model
        pretrained_dict = {k:v for k, v in checkpoint['model_state_dict'].items() if k in model_dict}

        # overwrite the entries in the existing state dictionary
        model_dict.update(pretrained_dict)
        
        # load the new state dict
        self.load_state_dict(model_dict)     

        if "use_speaker_embeds" in checkpoint.keys():
            self.use_speaker_embeds = checkpoint['use_speaker_embeds']
            
    def validate_model(self, dataloader, checkpoint=None):

        if checkpoint:
            self.load_model_from_dict(checkpoint)
        
        self.eval()
        embeds_vec, label_vec, accuracy_vec = [], [], []
        
        for i, data in enumerate(dataloader):                                
            embeds, recon_x, output = self.forward(data['features'].to(self.device), data['speaker_embeds'].to(self.device))
            
            accuracy = self.accuracy(output, data['labels'].to(self.device))
            
            embeds_vec.append(embeds.detach().cpu().numpy())
            label_vec.append(data['labels'].cpu().numpy())
            accuracy_vec.append(accuracy)
            
        reducer = umap.UMAP()
        umap_embeds = reducer.fit_transform(np.concatenate(embeds_vec, axis=0))
        
        print("Emotion prediction accuracy:{}".format(np.mean(accuracy_vec)))
        plt.figure()
        plt.scatter(umap_embeds[:, 0], umap_embeds[:, 1], c=np.concatenate(label_vec, axis=0), cmap='viridis')
        plt.pause(1) 

if __name__ == "__main__":
    # The configuration file
    config = yaml.load(open('src/config.yaml'), Loader=yaml.SafeLoader) 

    data = dd.io.load(str(Path(__file__).parents[2] / config['const_40mel_simple']))
    model = EmotionNet_AE(n_mels=40)
    
    embed, y, emotions = model.forward(data['features'][0:5].reshape(5, 40, -1), data['speaker_embeds'][0:5])
    print(embed.shape, y.shape, emotions.shape, data['labels'][0:5].shape)
    loss = model.loss_fn(data['features'][0:5], y, emotions, data['labels'][0:5])
    
    accuracy = model.accuracy(emotions, data['labels'][0:5])
    print(accuracy)
    
    model.griffin_lim_aud(data['features'][0].cpu().numpy())