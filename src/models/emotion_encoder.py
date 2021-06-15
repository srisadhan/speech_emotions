from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from yaml import safe_load
import os
import sys

from src.datasets.data_utils import split_audio_ixs, mel_spectrogram, preprocess_aud
from src.resemblyzer.voice_encoder import VoiceEncoder

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
np.random.seed(42)
import deepdish as dd
from pathlib import Path
import yaml
import umap 

# 1d Conv encoder for emotion embeddings that takes (N_mels x utterance_length) as input and has 256 latent space embeddings 
class EmotionEncoder_CONV(nn.Module):
    
    def __init__(self,
                 n_mels,
                 load_model=False,
                 epoch=1,
                 device=torch.device('cpu'),
                 use_speaker_embeds=False):
        
        super(EmotionEncoder_CONV, self).__init__()
        with open('src/config.yaml', 'r') as f:
            self.config = safe_load(f.read())

        self.device = device
        self.n_mels = n_mels
        self.use_speaker_embeds = use_speaker_embeds
        if self.use_speaker_embeds:
            self.final_embedding_size = 2 * self.config['embedding_size']
        else:
            self.final_embedding_size = self.config['embedding_size']

        self.encoder = nn.Sequential(self.conv_block(self.n_mels, 32, 32, 1, 0),
                                    # nn.MaxPool1d(kernel_size=8, stride=2),
                                    self.conv_block(32, 64, 16, 1, 0),
                                    # # nn.MaxPool1d(kernel_size=8, stride=2),
                                    self.conv_block(64, 128, 8, 1, 0),
                                    # # nn.MaxPool1d(kernel_size=8, stride=2),
                                    self.conv_block(128, self.config['embedding_size'], 8, 1, 0),
                                    nn.MaxPool1d(kernel_size=128)
                                    # nn.AvgPool1d(kernel_size=128)
                                    )

        self.weight_init()  #call before more nn definitions

        # self.linear1 = nn.Linear(self.final_embedding_size, self.config['n_emotions'])
    
       
        self.load_model = load_model
        self.epoch = epoch
        self.optimizer = None
        self.config = dict(self.config)

        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.]))

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()
        
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

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, nn.Conv1d):
                    torch.nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.fill_(0)

    def do_gradient_ops(self):
        # Gradient scale
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
            
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), 3, norm_type=2)

    def forward(self, x, speaker_embeds):
        # AutoEncoder
        embeds  = self.encoder(x)
        
        embeds = embeds.view(-1, self.config['embedding_size'])
        embeds = embeds.clone() / (torch.norm(embeds, dim=1, keepdim=True) + 1e-5)
        # concatenate the speaker embeddings to the encoder embeddings
        if self.use_speaker_embeds:
            embeds = torch.cat((embeds, speaker_embeds.squeeze()), dim=1)

        # A linear layer with ReLu activation to classifiy the emotions
        # output = F.relu(self.linear1(embeds))
        
        # return embeds, output
        return embeds


    def similarity_matrix(self, embeds):
        """ Computes the similarity matrix for Generalized-End-To-End-Loss 
        embeds : Embedding tensor of shape (emotions_per_batch, utterances_per_emotion, embedding_size)
        function used from:
        https://github.com/CorentinJ/Real-Time-Voice-Cloning/blob/95adc699c1deb637f485e85a5107d40da0ad94fc/encoder/model.py#L33
        """
        emotions_per_batch, utterances_per_emotion = embeds.shape[:2] 
        
        # centroid inclusive (eq. 1)
        # (Cj / |Cj|) gives the unit vector which is later used for finding cosine similarity 
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5) 
        
        
        # centroid exclusive (eq. 8)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_emotion - 1)
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

        # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
        # product of these vectors (which is just an element-wise multiplication reduced by a sum).
        # We vectorize the computation for efficiency.
        sim_matrix = torch.zeros(emotions_per_batch, utterances_per_emotion,
                                 emotions_per_batch).to(self.device)
        mask_matrix = 1 - np.eye(emotions_per_batch, dtype=np.int)
        for j in range(emotions_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias

        return sim_matrix
        
    def total_loss(self, embeds, labels):
        """
        Computes the softmax loss according the section 2.1 of Generalized End-To-End loss.
        
        embeds: the embeddings as a tensor of shape (emotions_per_batch, 
        utterances_per_emotion, embedding_size)
        """
        emotions_per_batch, utterances_per_emotion = embeds.shape[:2]
        
        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((emotions_per_batch * utterances_per_emotion, 
                                         emotions_per_batch))
        ground_truth = np.repeat(np.arange(emotions_per_batch), utterances_per_emotion)
        target = torch.from_numpy(ground_truth).long().to(self.device)
        GE2E_loss = self.loss_fn(sim_matrix, target)

        # EER (not backpropagated)
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, emotions_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            # Snippet from https://yangcha.github.io/EER-ROC/
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return GE2E_loss, eer
    
    def accuracy(self, output, labels):
        predictions = torch.argmax(output, dim=1)   
        correct = torch.sum(torch.eq(predictions, labels))

        accuracy = 100 * correct / labels.shape[0]
        
        return accuracy.detach().cpu().numpy()
      
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

    def train_model(self,
                   train_dataloader,
                   valid_dataloader,
                   emotions_per_batch, 
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


        if load_model:
            self.load_model_from_dict(checkpoint=checkpoint)

        for epoch in range(self.epoch, self.config['train_epochs']):
            self.epoch = epoch
            
            # set the model to training mode
            self.train(mode=True)
            loss_vec, eer_vec = [], []
            embeds_vec, label_vec = [], []
            for _, data in enumerate(train_dataloader):
                utterances_per_emotion = data['features'].shape[0]
                features = data['features'].reshape(emotions_per_batch * utterances_per_emotion, self.n_mels, -1).to(self.device)
                speaker_embeds = data['speaker_embeds'].reshape(emotions_per_batch * utterances_per_emotion, 1, self.config['embedding_size']).to(self.device)
                labels = data['labels'].reshape(emotions_per_batch * utterances_per_emotion).to(self.device)
                
                embeds = self.forward(features, speaker_embeds)
                loss, eer = self.total_loss(embeds.reshape(emotions_per_batch, utterances_per_emotion, self.final_embedding_size), labels)

                optimizer.zero_grad()
                loss.backward()
                # print(self.similarity_weight.grad, self.similarity_weight.is_leaf)
                self.do_gradient_ops()
                optimizer.step()
                
                # accuracy = self.accuracy(output, labels)
                
                loss_vec.append(loss.data.item())
                eer_vec.append(eer)
                embeds_vec.append(embeds.detach().cpu().numpy())
                label_vec.append(labels.cpu().numpy())
                
            eer_vec_valid, loss_vec_valid, embeds_vec_valid, label_vec_valid = [], [], [], []
            for i, data_valid in enumerate(valid_dataloader): 
                utterances_per_emotion = data_valid['features'].shape[0]
                features = data_valid['features'].reshape(emotions_per_batch * utterances_per_emotion, self.n_mels, -1).to(self.device)
                speaker_embeds = data_valid['speaker_embeds'].reshape(emotions_per_batch * utterances_per_emotion, 1, self.config['embedding_size']).to(self.device)
                labels = data_valid['labels'].reshape(emotions_per_batch * utterances_per_emotion).to(self.device)
                 
                embeds = self.forward(features, speaker_embeds)
                # accuracy = self.accuracy(output, data_valid['labels'].to(self.device))
                loss_valid, eer_valid = self.total_loss(embeds.reshape(emotions_per_batch, utterances_per_emotion, self.final_embedding_size), labels)
                
                loss_vec_valid.append(loss_valid.data.item())
                eer_vec_valid.append(eer_valid)
                embeds_vec_valid.append(embeds.detach().cpu().numpy())
                label_vec_valid.append(labels.cpu().numpy())

            self.writer.add_scalars('Running loss', {'Training':np.mean(loss_vec),
                                                    'Validation':np.mean(loss_vec_valid)}, epoch)
            self.writer.add_scalars('EER', {'Training':np.mean(eer_vec),
                                                'Validation':np.mean(eer_vec_valid)}, epoch)

            if lr_scheduler:
                if epoch % 500 == 0:
                    lr_scheduler.step()
                
            if epoch % 5000 == 0:
                print("Device: {}, Epoch: {}, Loss: {}, EER- train:{}, Validation:{}".format(self.device, epoch, np.mean(loss_vec), np.mean(eer_vec), np.mean(eer_vec_valid)))
                                
                torch.save(
                    {
                        'epoch': self.epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'use_speaker_embeds': self.use_speaker_embeds,
                    }, self.model_save_string.format(epoch))
            
                reducer = umap.UMAP()
                umap_embeds_train = reducer.fit_transform(np.concatenate(embeds_vec, axis=0))
                umap_embeds_valid = reducer.fit_transform(np.concatenate(embeds_vec_valid, axis=0))

                _, ax = plt.subplots(1,2)
                ax[0].scatter(umap_embeds_train[:, 0], umap_embeds_train[:, 1], c=np.concatenate(label_vec, axis=0), cmap='viridis')
                ax[0].set_title('Training data embeddings')
                ax[1].scatter(umap_embeds_valid[:, 0], umap_embeds_valid[:, 1], c=np.concatenate(label_vec_valid, axis=0), cmap='viridis')
                ax[1].set_title('Validation data embeddings')
                savepath = os.path.join(self.config['vis_dir'], 'Mel_{}_spk_{}'.format(str(self.n_mels), self.use_speaker_embeds))
                os.makedirs(savepath, exist_ok=True)

                plt.savefig(os.path.join(savepath, 'epoch_{}.png'.format(self.epoch)))
                            
    def validate_model(self, dataloader, checkpoint=None):
        if checkpoint:
            self.load_model_from_dict(checkpoint)
        
        self.eval()
        embeds_vec, label_vec, eer_vec, loss_vec = [], [], [], []
        
        for i, data in enumerate(dataloader):      
            utterances_per_emotion = data['features'].shape[0]
            emotions_per_batch = data['features'].shape[1]
            features = data['features'].reshape(emotions_per_batch * utterances_per_emotion, self.n_mels, -1).to(self.device)
            speaker_embeds = data['speaker_embeds'].reshape(emotions_per_batch * utterances_per_emotion, 1, self.config['embedding_size']).to(self.device)
            labels = data['labels'].reshape(emotions_per_batch * utterances_per_emotion).to(self.device)
                       
            embeds = self.forward(features, speaker_embeds)
            loss, eer = self.total_loss(embeds.reshape(emotions_per_batch, utterances_per_emotion, self.final_embedding_size), labels)
                
            # accuracy = self.accuracy(output, data['labels'].to(self.device))
            
            embeds_vec.append(embeds.detach().cpu().numpy())
            label_vec.append(labels.cpu().numpy())
            eer_vec.append(eer)
            loss_vec.append(loss.data.item())

        reducer = umap.UMAP()
        umap_embeds = reducer.fit_transform(np.concatenate(embeds_vec, axis=0))
        
        print("Emotion prediction EER:{}, loss:{}".format(np.mean(eer_vec), np.mean(loss_vec)))
        plt.figure()
        plt.scatter(umap_embeds[:, 0], umap_embeds[:, 1], c=np.concatenate(label_vec, axis=0), cmap='viridis')
        savepath = os.path.join(self.config['vis_dir'], 'Mel_{}_spk_{}'.format(str(self.n_mels), self.use_speaker_embeds))
        os.makedirs(savepath, exist_ok=True)

        plt.savefig(os.path.join(savepath, 'Test_data_{}.png'.format(self.epoch)))

# 1d Conv encoder for emotion embeddings that takes (N_mels x utterance_length) as input and has 256 latent space embeddings 
class EmotionEncoder_LSTM(nn.Module):
    
    def __init__(self,
                 n_mels,
                 load_model=False,
                 epoch=1,
                 device=torch.device('cpu'),
                 use_speaker_embeds=False):
        
        super(EmotionEncoder_LSTM, self).__init__()
        with open('src/config.yaml', 'r') as f:
            self.config = safe_load(f.read())

        self.device = device
        self.n_mels = n_mels
        self.hidden_size = self.config['hidden_size']
        self.use_speaker_embeds = use_speaker_embeds
        
        if self.use_speaker_embeds:
            self.final_embedding_size = 2 * self.config['embedding_size']
        else:
            self.final_embedding_size = self.config['embedding_size']

        self.lstm = nn.LSTM(input_size=self.n_mels,
                            hidden_size=self.config['hidden_size'], 
                            num_layers=self.config['num_layers'], 
                            batch_first=True)
        self.linear = nn.Linear(in_features=self.config['hidden_size'], 
                                out_features=self.config['embedding_size'])

       
        self.load_model = load_model
        self.epoch = epoch
        self.optimizer = None
        self.config = dict(self.config)

        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.]))

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        # load speaker voice encoder 
        self.VoiceEncoder = VoiceEncoder()
        
    def do_gradient_ops(self):
        # Gradient scale
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
            
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), 3, norm_type=2)

    def forward(self, x, speaker_embeds=None, hidden_init=None):
        out, (hidden, cell) = self.lstm(x, hidden_init)
        
        # We take only the hidden state of the last layer
        embeds = F.relu(self.linear(hidden[-1]))
        
        embeds = embeds.view(-1, self.config['embedding_size'])
        embeds = embeds.clone() / (torch.norm(embeds, dim=1, keepdim=True) + 1e-5)
        # concatenate the speaker embeddings to the encoder embeddings
        if self.use_speaker_embeds:
            embeds = torch.cat((embeds, speaker_embeds.reshape(-1, self.config['embedding_size'])), dim=1)

        return embeds

    def compute_partial_slices(self, aud):
        """ preprocess the audio clip, extract mel-spectrogram and speaker embeddings and then split it into samples of equal size
        """
        # aud, sample_rate = preprocess_aud(aud) # already preprocessed the data while loading it in load_audio_RAVDESS function
        aud_splits, mel_splits = split_audio_ixs(len(aud))
        max_aud_length = aud_splits[-1].stop
        if max_aud_length >= len(aud):
            aud = np.pad(aud, (0, max_aud_length - len(aud)), "constant")

        specgram = mel_spectrogram(aud, sr=self.config['resampled_rate']).T
        specgrams = np.array([specgram[s] for s in mel_splits])
        
        # create the embedding using Voice Encoder for each partial utterance
        embeds = np.array([self.VoiceEncoder.embed_utterance(aud[s]).reshape(1, -1) for s in aud_splits])
        return specgrams, embeds
    
    def embed_emotion(self, aud):
        """ Compute the partial embeddings and average them to obtain the emotion embedding for the utterance
        """
        specgrams, spk_embeds = self.compute_partial_slices(aud)
        
        with torch.no_grad():
            specgrams = torch.from_numpy(specgrams).to(self.device)
            spk_embeds = torch.from_numpy(spk_embeds).to(self.device)
            partial_embeds = self.forward(specgrams, spk_embeds).cpu().numpy()
            emotion_embedding = np.mean(partial_embeds, axis=0)
            emotion_embedding = emotion_embedding #/ (np.linalg.norm(emotion_embedding, 2) + 1e-5)
        
        return emotion_embedding
              
    def similarity_matrix(self, embeds):
        """ Computes the similarity matrix for Generalized-End-To-End-Loss 
        embeds : Embedding tensor of shape (emotions_per_batch, utterances_per_emotion, embedding_size)
        function used from:
        https://github.com/CorentinJ/Real-Time-Voice-Cloning/blob/95adc699c1deb637f485e85a5107d40da0ad94fc/encoder/model.py#L33
        """
        emotions_per_batch, utterances_per_emotion = embeds.shape[:2] 
        
        # centroid inclusive (eq. 1)
        # (Cj / |Cj|) gives the unit vector which is later used for finding cosine similarity 
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5) 
        
        
        # centroid exclusive (eq. 8)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_emotion - 1)
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

        # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
        # product of these vectors (which is just an element-wise multiplication reduced by a sum).
        # We vectorize the computation for efficiency.
        sim_matrix = torch.zeros(emotions_per_batch, utterances_per_emotion,
                                 emotions_per_batch).to(self.device)
        mask_matrix = 1 - np.eye(emotions_per_batch, dtype=np.int)
        for j in range(emotions_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias

        return sim_matrix
        
    def total_loss(self, embeds, labels):
        """
        Computes the softmax loss according the section 2.1 of Generalized End-To-End loss.
        
        embeds: the embeddings as a tensor of shape (emotions_per_batch, 
        utterances_per_emotion, embedding_size)
        """
        emotions_per_batch, utterances_per_emotion = embeds.shape[:2]
        
        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((emotions_per_batch * utterances_per_emotion, 
                                         emotions_per_batch))
        ground_truth = np.repeat(np.arange(emotions_per_batch), utterances_per_emotion)
        target = torch.from_numpy(ground_truth).long().to(self.device)
        GE2E_loss = self.loss_fn(sim_matrix, target)

        # EER (not backpropagated)
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, emotions_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            # Snippet from https://yangcha.github.io/EER-ROC/
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return GE2E_loss, eer
    
    def accuracy(self, output, labels):
        predictions = torch.argmax(output, dim=1)   
        correct = torch.sum(torch.eq(predictions, labels))

        accuracy = 100 * correct / labels.shape[0]
        
        return accuracy.detach().cpu().numpy()
      
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

    def sort_data(self, data, emotions_per_batch, utterances_per_emotion):
        labels = data['labels'].reshape(emotions_per_batch * utterances_per_emotion).numpy()
        sorted_lables = np.sort(labels)
        plt.figure()
        plt.plot(sorted_lables, 'r.')
        plt.savefig('temp1.png')
        plt.pause(1)
        # .to(self.device)
        # features = data['features'].reshape(emotions_per_batch * utterances_per_emotion, -1, self.n_mels).to(self.device)
        # speaker_embeds = data['speaker_embeds'].reshape(emotions_per_batch * utterances_per_emotion, 1, self.config['embedding_size']).to(self.device)
                    
    def train_model(self,
                   train_dataloader,
                   valid_dataloader,
                   emotions_per_batch, 
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
                                      model_log_dir, "mel_{}_spk_{}_run_{}".format(self.n_mels, self.use_speaker_embeds,
                                      len(os.listdir(model_log_dir)) if os.path.exists(model_log_dir) else 0))
                                      )
            self.model_save_string = os.path.join(
                    model_save_dir, self.__class__.__name__ + '_Epoch_{}.pt')
            
            os.makedirs(model_save_dir, exist_ok=True)
            os.makedirs(self.config['vis_dir'], exist_ok=True)
            
            self.writer = SummaryWriter(log_dir=os.path.join(
                                        run_log_dir, "mel_{}_spk_{}_run_{}".format(self.n_mels, self.use_speaker_embeds,
                                        len(os.listdir(run_log_dir)) if os.path.exists(run_log_dir) else 0)))
        else:
            model_save_dir = os.path.join(os.path.join(
                                      model_log_dir, "mel_{}_run_{}".format(self.n_mels,
                                      len(os.listdir(model_log_dir)) - 1 if os.path.exists(model_log_dir) else 0))
                                      )
            self.model_save_string = os.path.join(
                model_save_dir, self.__class__.__name__ + '_Epoch_{}.pt')       


        if load_model:
            self.load_model_from_dict(checkpoint=checkpoint)

        for epoch in range(self.epoch, self.config['train_epochs']):
            self.epoch = epoch
            
            # set the model to training mode
            self.train(mode=True)
            loss_vec, eer_vec = [], []
            embeds_vec, label_vec = [], []
            for _, data in enumerate(train_dataloader):
                emotions_per_batch = data['features'].squeeze().shape[0]
                utterances_per_emotion = data['features'].squeeze().shape[1]

                features = data['features'].squeeze().reshape(emotions_per_batch * utterances_per_emotion, -1, self.n_mels).to(self.device)
                speaker_embeds = data['speaker_embeds'].squeeze().reshape(emotions_per_batch * utterances_per_emotion, 1, self.config['embedding_size']).to(self.device)
                labels = data['labels'].squeeze().reshape(emotions_per_batch * utterances_per_emotion).to(self.device)
            
                embeds = self.forward(features, speaker_embeds)
                loss, eer = self.total_loss(embeds.reshape(emotions_per_batch, utterances_per_emotion, self.final_embedding_size), labels)

                optimizer.zero_grad()
                loss.backward()
                self.do_gradient_ops()
                optimizer.step()
                                
                loss_vec.append(loss.data.item())
                eer_vec.append(eer)
                embeds_vec.append(embeds.detach().cpu().numpy())
                label_vec.append(labels.cpu().numpy())
                
            eer_vec_valid, loss_vec_valid, embeds_vec_valid, label_vec_valid = [], [], [], []
            for i, data_valid in enumerate(valid_dataloader): 
                emotions_per_batch = data_valid['features'].squeeze().shape[0]
                utterances_per_emotion = data_valid['features'].squeeze().shape[1]

                features = data_valid['features'].squeeze().reshape(emotions_per_batch * utterances_per_emotion, -1, self.n_mels).to(self.device)
                speaker_embeds = data_valid['speaker_embeds'].squeeze().reshape(emotions_per_batch * utterances_per_emotion, 1, self.config['embedding_size']).to(self.device)
                labels = data_valid['labels'].squeeze().reshape(emotions_per_batch * utterances_per_emotion).to(self.device)
                 
                valid_embeds = self.forward(features, speaker_embeds)
                # accuracy = self.accuracy(output, data_valid['labels'].to(self.device))
                loss_valid, eer_valid = self.total_loss(valid_embeds.reshape(emotions_per_batch, utterances_per_emotion, self.final_embedding_size), labels)
                
                loss_vec_valid.append(loss_valid.data.item())
                eer_vec_valid.append(eer_valid)
                embeds_vec_valid.append(valid_embeds.detach().cpu().numpy())
                label_vec_valid.append(labels.cpu().numpy())

            self.writer.add_scalars('Running loss', {'Training':np.mean(loss_vec),
                                                    'Validation':np.mean(loss_vec_valid)}, epoch)
            self.writer.add_scalars('EER', {'Training':np.mean(eer_vec),
                                                'Validation':np.mean(eer_vec_valid)}, epoch)

            if lr_scheduler:
                if epoch % 50 == 0:
                    lr_scheduler.step()
                
            if epoch % 1000 == 0:
                print("Device: {}, Epoch: {}, Loss: {}, EER- train:{}, Validation:{}".format(self.device, epoch, np.mean(loss_vec), np.mean(eer_vec), np.mean(eer_vec_valid)))
                                
                torch.save(
                    {
                        'epoch': self.epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'use_speaker_embeds': self.use_speaker_embeds,
                    }, self.model_save_string.format(epoch))
            
                reducer = umap.UMAP()
                umap_embeds_train = reducer.fit_transform(np.concatenate(embeds_vec, axis=0))
                umap_embeds_valid = reducer.fit_transform(np.concatenate(embeds_vec_valid, axis=0))

                _, ax = plt.subplots(1,2)
                ax[0].scatter(umap_embeds_train[:, 0], umap_embeds_train[:, 1], c=np.concatenate(label_vec, axis=0), cmap='viridis')
                ax[0].set_title('Training data embeddings')
                ax[1].scatter(umap_embeds_valid[:, 0], umap_embeds_valid[:, 1], c=np.concatenate(label_vec_valid, axis=0), cmap='viridis')
                ax[1].set_title('Validation data embeddings')
                savepath = os.path.join(self.config['vis_dir'], 'Mel_{}_spk_{}'.format(str(self.n_mels), self.use_speaker_embeds))
                os.makedirs(savepath, exist_ok=True)

                plt.savefig(os.path.join(savepath, 'epoch_{}.png'.format(self.epoch)))
                            
    def validate_model(self, dataloader, checkpoint=None, savefig=True):
        if checkpoint:
            self.load_model_from_dict(checkpoint)
        
        self.eval()
        embeds_vec, label_vec, eer_vec, loss_vec = [], [], [], []
        
        for i, data in enumerate(dataloader):      
            emotions_per_batch = data['features'].squeeze().shape[0]
            utterances_per_emotion = data['features'].squeeze().shape[1]

            features = data['features'].squeeze().reshape(emotions_per_batch * utterances_per_emotion, -1, self.n_mels).to(self.device)
            speaker_embeds = data['speaker_embeds'].squeeze().reshape(emotions_per_batch * utterances_per_emotion, 1, self.config['embedding_size']).to(self.device)
            labels = data['labels'].squeeze().reshape(emotions_per_batch * utterances_per_emotion).to(self.device)
                           
            embeds = self.forward(features, speaker_embeds)
            loss, eer = self.total_loss(embeds.reshape(emotions_per_batch, utterances_per_emotion, self.final_embedding_size), labels)
                
            # accuracy = self.accuracy(output, data['labels'].to(self.device))
            
            embeds_vec.append(embeds.detach().cpu().numpy())
            label_vec.append(labels.cpu().numpy())
            eer_vec.append(eer)
            loss_vec.append(loss.data.item())

        reducer = umap.UMAP()
        umap_embeds = reducer.fit_transform(np.concatenate(embeds_vec, axis=0))
        
        print("Emotion prediction EER:{}, loss:{}".format(np.mean(eer_vec), np.mean(loss_vec)))
        plt.figure()
        plt.scatter(umap_embeds[:, 0], umap_embeds[:, 1], c=np.concatenate(label_vec, axis=0), cmap='viridis')
        savepath = os.path.join(self.config['vis_dir'], 'Mel_{}_spk_{}'.format(str(self.n_mels), self.use_speaker_embeds))
        os.makedirs(savepath, exist_ok=True)

        if savefig:
            plt.savefig(os.path.join(savepath, 'Test_data_{}.png'.format(self.epoch)))


if __name__ == "__main__":
    # The configuration file
    config = yaml.load(open('src/config.yaml'), Loader=yaml.SafeLoader) 

    data = dd.io.load(str(Path(__file__).parents[2] / config['const_40mel_simple']))
    model = EmotionEncoder_CONV(n_mels=40, use_speaker_embeds=True)
    
    embed = model.forward(data['features'][0:5].reshape(5, 40, -1), data['speaker_embeds'][0:5])
    print(embed.shape, data['labels'][0:5].shape)
    
    loss, eer = model.total_loss(data['features'][0:5], data['labels'][0:5])
    
    print( eer)

    loss.backward()

    # accuracy = model.accuracy(output, data['labels'][0:5])
    # print(accuracy)
    