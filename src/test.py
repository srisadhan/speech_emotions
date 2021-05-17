import torch
import torch.nn as nn

def reparameterize(mu, logvar):
    mu  = mu
    std = torch.exp(torch.div(logvar, 2))
    eps = torch.empty(std.shape).normal_(0.0, 1.0)
    z = mu + (eps * std)

    return z

##########################################
### Working part ###
##########################################  
inp = torch.rand(64, 80, 248)

## --------Encoder---------- ##
enc_module = nn.Sequential(nn.Conv1d(80, 124, 4, 2, 1, 1), nn.ReLU(True),
                       nn.Conv1d(124, 62, 4, 2, 1), nn.ReLU(True), 
                       nn.Conv1d(62, 32, 4, 2, 1), nn.ReLU(True), 
                       nn.Conv1d(32, 16, 4, 2, 1), nn.ReLU(True), 
                       nn.Conv1d(16, 16, 4, 2, 1), nn.ReLU(True),                     
                       nn.Conv1d(16, 128, 4, 2, 1), nn.ReLU(True),
                       nn.Conv1d(128, 256, 3, 2), nn.ReLU(True),                       
                       )

inter_linear = nn.Linear(256, 16)

## --------Decoder---------- ##
invert_linear = nn.Linear(8, 256)
        
dec_module = nn.Sequential(nn.ReLU(True), nn.ConvTranspose1d(256, 128, 3),
                            nn.ReLU(True), nn.ConvTranspose1d(128, 16, 4, 2, 1),
                            nn.ReLU(True), nn.ConvTranspose1d(16, 16, 4, 2),
                            nn.ReLU(True), nn.ConvTranspose1d(16, 32, 4, 2),
                            nn.ReLU(True), nn.ConvTranspose1d(32, 62, 4, 2),
                            nn.ReLU(True), nn.ConvTranspose1d(62, 124, 4, 2, 1),
                            nn.ReLU(True), nn.ConvTranspose1d(124, 80, 4, 2, 1),
                            )

x = inter_linear(enc_module(inp).view(-1, 256))
x_mu     = x[:, :8]
x_logvar = x[:, 8:]

# reparameterization trick
z = reparameterize(x_mu, x_logvar)

temp = invert_linear(z).unsqueeze(-1) # similar to view(-1, 256, 1)
out  = dec_module(temp)

print('New module: ', inp.size(), out.size())

inp = torch.rand(64, 80, 128)

## --------Encoder---------- ##
enc_module = nn.Sequential(nn.Conv1d(80, 64, 4, 2, 1, 1), nn.ReLU(True),
                       nn.Conv1d(64, 32, 4, 2, 1), nn.ReLU(True), 
                       nn.Conv1d(32, 16, 4, 2, 1), nn.ReLU(True), 
                       nn.Conv1d(16, 16, 4, 2, 1), nn.ReLU(True),                     
                       nn.Conv1d(16, 128, 4, 2, 1), nn.ReLU(True),
                       nn.Conv1d(128, 256, 3, 2), nn.ReLU(True),                       
                       )

inter_linear = nn.Linear(256, 16)

## --------Decoder---------- ##
invert_linear = nn.Linear(8, 256)
        
dec_module = nn.Sequential(nn.ReLU(True), nn.ConvTranspose1d(256, 128, 3),
                            nn.ReLU(True), nn.ConvTranspose1d(128, 16, 4, 2),
                            nn.ReLU(True), nn.ConvTranspose1d(16, 16, 4, 2, 1),
                            nn.ReLU(True), nn.ConvTranspose1d(16, 32, 4, 2, 1),
                            nn.ReLU(True), nn.ConvTranspose1d(32, 64, 4, 2, 1),
                            nn.ReLU(True), nn.ConvTranspose1d(64, 80, 4, 2, 1),
                            )

x = inter_linear(enc_module(inp).view(-1, 256))
print('X-size: ', x.size())
x_mu     = x[:, :8]
x_logvar = x[:, 8:]

# reparameterization trick
z = reparameterize(x_mu, x_logvar)

temp = invert_linear(z).unsqueeze(-1) # similar to view(-1, 256, 1)
out  = dec_module(temp)

print('New module: ', inp.size(), out.size())
