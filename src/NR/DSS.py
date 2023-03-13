import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, C, L, N):
        super(Encoder, self).__init__()
        self.C = C # in_channels
        self.L = L  # length_kernel
        self.N = N  # n_output
        self.conv = nn.Conv2d(in_channels=C,
                                out_channels=N,
                                kernel_size=(L,1),
                                stride=(L,1),
                                padding=0,
                                bias=False)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class RNN(nn.Module):
    def __init__(self, C, hidden_size= 1024,  num_layers=2):
        super(RNN, self).__init__()
        
        self.rnn = nn.GRU(input_size = C, hidden_size = hidden_size, num_layers = num_layers, batch_first  = True, bidirectional  = False)
        self.activation = nn.PReLU()
        self.FC = nn.Linear(hidden_size,C)

    def forward(self, x, h = None):
        x,h_out = self.rnn(x,h)
        x = self.activation(x)
        x = self.FC(x)
        return x,h_out

class DSS_v1(nn.Module):
    def __init__(self, C=5, ch = 4):
        super(DSS_v1, self).__init__()
        
        self.n_fft = 512
        self.n_hfft = 257
        dim_feature = 1024
        self.ch = ch

        # Convolution Encoders
        self.encoders = []
        enc = Encoder(C=C,N=8,L=2)
        self.encoders.append(enc)
        self.add_module("enc_{}".format(0),enc)
        for i in range(1,8):
            enc = Encoder(C=2**(i+2),N=2**(i+3),L=2)
            self.encoders.append(enc)
            self.add_module("enc_{}".format(i),enc)

    
        # Transformer Encoder
        self.formers = []
        self.formers.append(nn.TransformerEncoderLayer(d_model=dim_feature,nhead=8))
        self.formers.append(nn.TransformerEncoderLayer(d_model=dim_feature,nhead=8))

        for i,former in enumerate(self.formers) : 
            self.add_module("former_{}".format(i),former)
        
        self.recurrents = []
        self.recurrents.append(RNN(dim_feature))
        self.recurrents.append(RNN(dim_feature))

        for i,rnn in enumerate(self.recurrents) : 
            self.add_module("rnn_{}".format(i),rnn)
        
        self.FC= nn.Linear(dim_feature,2*self.n_hfft*ch)
        self.last = nn.Tanh()
        self.add_module("FC_{}".format(0),self.FC)


        ## For inference
        self.window = torch.hann_window(self.n_fft)

    def forward(self,x):
        for enc in self.encoders : 
            x = enc(x)
        
        # x : [B, 1024,1,T]
        x = torch.reshape(x,(x.shape[0],x.shape[1],x.shape[3]))
        x = torch.permute(x,(0,2,1))

        for former in self.formers : 
            x = former(x)
        
        for recurrent in self.recurrents : 
            x,h = recurrent(x)
        
        w = self.FC(x)
        w = self.last(w)
        w = torch.permute(w,(0,2,1))
        w = torch.reshape(w,(w.shape[0],2*self.ch,self.n_hfft,w.shape[-1]))
        w = w[:,0::2,:,:] + 1j*w[:,1::2,:,:]
        return w

    def output(self, noisy,filter,hp=None) : 
        self.window = self.window.to(filter.get_device())

        B,C,L = noisy.shape

        noisy = torch.reshape(noisy,(B*C,L))

        B,C,F,T = filter.shape

        spec = torch.stft(noisy,n_fft=self.n_fft,window=self.window,return_complex=True)[...,:500]

        spec = torch.reshape(spec,(B,C,F,T))

        spec = torch.permute(spec,(0,2,3,1))
        filter = torch.permute(filter,(0,2,3,1))

        spec = torch.reshape(spec,(B*F*T,1,C))
        filter = torch.reshape(filter,(B*F*T,C,1))

        # [B*F*T,1,1]
        output = torch.bmm(spec,filter)
        output = torch.reshape(output,(B,F,T))

        output = torch.istft(output,n_fft=512,window=self.window,length=L)

        return output