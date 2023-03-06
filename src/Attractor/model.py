import torch
import torch.nn as nn

"""
Direction Attractor Net

Y. Nakagome, M. Togami, T. Ogawa and T. Kobayashi, "Deep Speech Extraction with Time-Varying Spatial Filtering Guided By Desired Direction Attractor," ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 671-675, doi: 10.1109/ICASSP40776.2020.9053629.

"""
class encoder(nn.Module):
    def __init__(self,
                 n_channel=4,
                 n_fft=512,
                 n_dim=1024,
                 n_hidden_layer=2,
                 n_latent = 40,
                 type_activation=None,
                 type_normalization=None
                 ):
        super(encoder, self).__init__()
        n_hfft = n_fft//2+1
        n_dim = 1024
        n_dim_in = n_channel * n_hfft

        if type_activation == "PReLU" : 
            activation = nn.PReLU
        elif type_activation == "ReLU" : 
            activation = nn.ReLU
        else :
            activation = nn.Identity

        if type_normalization == "BatchNorm2d":
            normalization = nn.BatchNorm2d
        else  :
            normalization = nn.Identity

        
        self.layers=[]

        module = nn.Sequential(
            nn.Linear(n_dim_in,n_dim),
            activation(),
            normalization(n_dim)
        )
        self.add_module("layer_{}".format(0),module)
        self.layers.append(module)

        for i in range(n_hidden_layer):
            module = nn.Sequential(
                nn.Linear(n_dim,n_dim),
                activation(),
                normalization(n_dim)
                )
            self.add_module("layer_{}".format(i+1),module)
            self.layers.append(module)

        module = nn.Sequential(
            nn.Linear(n_dim,n_latent),
            activation(),
            normalization(n_latent)
            )
        self.add_module("layer_{}".format(n_hidden_layer+1),module)
        self.layers.append(module)


    def forward(self,x):
        for i,layer in enumerate(self.layers):
            x = layer(x)
        return x

class DirectionAttractor(nn.Module) : 
    def __init__(self,
                 n_channel=4,
                 n_fft=512,
                 n_dim=1024,
                 n_hidden_layer=2,
                 n_latent = 40,
                 type_activation=None,
                 type_normalization=None
                 ):
        super(DirectionAttractor, self).__init__()
        n_hfft = n_fft//2+1
        n_dim = 1024
        n_dim_in = n_channel * n_hfft

        self.ebedding_speech = encoder(
                n_channel=4,
                n_fft=512,
                n_dim=1024,
                n_hidden_layer=2,
                n_latent = 40,
                type_activation=None,
                type_normalization=None
        )  
        self.ebedding_angle = encoder(
                n_channel=4,
                n_fft=512,
                n_dim=1024,
                n_hidden_layer=2,
                n_latent = 40,
                type_activation=None,
                type_normalization=None
        )

        # TODO
        self.estimation_target_mask
        self.estimation_target_activity
        self.estimation_noise_mask
        self.estimation_noise_activity


    def forward(self,x):


class DirectionAttractorNet(nn.Module):
    def __init__(self,n_channel,n_fft=512):
        super(DirectionAttractorNet, self).__init__()

        self.n_channel = n_channel
        self.n_fft = n_fft
        self.n_hfft = n_fft//2+1

        self.window = torch.hann_window(self.n_fft)

    """
    x : [B, C, L], mutli-channel time-domain signal
    y : [B, 1, L], estimated time-domain signal
    """
    def forward(self,x):
        # reshape : 
        X = torch.stft(x,n_fft = self.n_fft,window=self.window)
        # reshape :

        # apply filter
        Y = X

        y = torch.istft(Y,n_fft = self.n_fft,window=self.window)
        # reshape
        return y