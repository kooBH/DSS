import torch
from torch import nn
import torch.nn.functional as F

try :
    from DPRNN import Encoder, Decoder, Dual_Path_RNN
except ImportError :
    from .DPRNN import Encoder, Decoder, Dual_Path_RNN

class Dual_RNN_model(nn.Module):
    def __init__(self, in_channels, out_channels, latent_encoder=256, latent_separation=64, hidden_channels=128,
                 kernel_size=2, rnn_type='LSTM', norm='ln', dropout=0,
                 bidirectional=False, num_layers=4, K=200, num_spks=1):
        super(Dual_RNN_model,self).__init__()

        self.encoder = Encoder(in_channels=in_channels,kernel_size=kernel_size,out_channels=latent_encoder)
        
        self.separation = Dual_Path_RNN(latent_encoder, latent_separation, hidden_channels,
                 rnn_type=rnn_type, norm=norm, dropout=dropout,
                 bidirectional=bidirectional, num_layers=num_layers, K=K, num_spks=num_spks)

        self.decoder = Decoder(in_channels=latent_encoder, out_channels=out_channels, kernel_size=kernel_size, stride=kernel_size//2, bias=False)
        self.num_spks = num_spks
    
    def forward(self, x,angle=None):
        '''
           x: [B, L]
        '''
        print(x.shape)
        # [B, N, L]
        e = self.encoder(x)
        print("e : {}".format(e.shape))

        # [spks, B, N, L]
        s = self.separation(e)
        print("s : {}".format(s.shape))
        # [B, N, L] -> [B, L]
        out = [s[i]*e for i in range(self.num_spks)]
        audio = [self.decoder(out[i]) for i in range(self.num_spks)]
        return audio
    

if __name__ == '__main__':
    model = Dual_RNN_model(4,1)
    x = torch.randn(2,4,16000)
    audio = model(x)

    import pdb
    pdb.set_trace()