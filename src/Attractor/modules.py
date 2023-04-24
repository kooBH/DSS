import torch
import torch.nn as nn

class permuteTF(nn.Module):
    def __init__(self):
        super(permuteTF, self).__init__()

    def forward(self,x):
        if len(x.shape) == 3 :
            x = torch.permute(x,(0,2,1))
        elif len(x.shape) == 4 :
            x = torch.permute(x,(0,1,3,2))
        return x

class encoder(nn.Module):
    def __init__(self,
                 n_in = 257,
                 n_dim=1024,
                 n_hidden_layer=3,
                 n_out = 40,
                 type_activation=None,
                 type_normalization=None,
                 dropout = 0.0
                 ):
        super(encoder, self).__init__()
        n_dim = 1024

        if type_activation == "PReLU" : 
            activation = nn.PReLU
        elif type_activation == "ReLU" : 
            activation = nn.ReLU
        else :
            activation = nn.Identity

        if type_normalization == "BatchNorm":
            normalization = nn.BatchNorm1d
        else  :
            normalization = nn.Identity

        self.encoder = []
        self.acitvation = []
        self.normalization = []

        self.layers = []
        module = nn.Sequential(
            nn.Linear(n_in,n_dim),
            activation(),
            permuteTF(),
            normalization(n_dim),
            permuteTF()
        )
        self.layers.append(module)

        for i in range(n_hidden_layer):
            module = nn.Sequential(
                nn.Linear(n_dim,n_dim),
                activation(),
                permuteTF(),
                normalization(n_dim),
                permuteTF()
            )
            self.layers.append(module)

        module = nn.Sequential(
                nn.Linear(n_dim,n_out),
                activation(),
                permuteTF(),
                normalization(n_out),
                permuteTF()
            )
        self.layers.append(module)

        self.layers = nn.ModuleList(self.layers)
        self.DR = nn.Dropout(dropout)

    def forward(self,x):
        for i in range(len(self.layers)) :
            x = self.layers[i](x)
            x = self.DR(x)
        return x
    
class estimator(nn.Module):
    def __init__(self,
                 n_in,
                 n_out,
                 type_activation="Sigmoid",
                 type_normalization=None
                 ) :
        super(estimator,self).__init__()

        if type_activation == "PReLU" : 
            activation = nn.PReLU
        elif type_activation == "ReLU" : 
            activation = nn.ReLU
        elif type_activation == "Sigmoid" : 
            activation = nn.Sigmoid
        else :
            activation = nn.Identity

        if type_normalization == "BatchNorm":
            normalization = nn.BatchNorm1d
        else  :
            normalization = nn.Identity

        self.layer = nn.Sequential(
            nn.Linear(n_in,n_out),
            activation(),
            normalization(n_out)
        )
    
    def forward(self,x):
        return self.layer(x)

class RNN_block(nn.Module) : 
    def __init__(self,dim_in,dim_out,
                 style="GRU"
                 ):
        super(RNN_block, self).__init__()

        if style == "GRU" : 
            self.layer = nn.GRU(dim_in,dim_out,batch_first=True)
        elif style == "LSTM" : 
            self.layer = nn.LSTM(dim_in,dim_out,batch_first=True)
        else : 
            self.layer = nn.RNN(dim_in,dim_out,batch_first=True)

        self.permuteTF = permuteTF()
        self.norm = nn.BatchNorm1d(dim_out)
        self.permuteFT = permuteTF()
        self.activation = nn.PReLU()

    def forward(self,x):
        x,h = self.layer(x)
        x = self.permuteTF(x)
        x = self.norm(x)
        x = self.permuteFT(x)
        x = self.activation(x)
        return x
    
class RNN(nn.Module) : 
    def __init__(self,dim_in,dim_out,
                 n_hidden=1024,
                 n_rnn_layer=2,
                 style="GRU"
                 ):
        super(RNN, self).__init__()

        if n_rnn_layer < 2 :
            Exception("n_rnn_layer must be larger than 2")

        self.layers = []
        self.layers.append(RNN_block(dim_in,n_hidden,style=style))
        for i in range(n_rnn_layer-2) : 
            self.layers.append(RNN_block(n_hidden,n_hidden,style=style))
        self.layers.append(RNN_block(n_hidden,dim_out,style=style))
        self.layers = nn.ModuleList(self.layers)

        # TODO : TGRU, FGRU

    def forward(self,x):
        for i in range(len(self.layers)) :
            x = self.layers[i](x)

        return x


class SpectralFeature(nn.Module):
    def __init__(self, 
                 list_feature = ["X","SV"],
                 n_fft = 512,
                 n_channel = 4,
                 ss = 340.4,
                 dist = 1,
                 sr=16000
                 
                 ) : 
        super(SpectralFeature,self).__init__()
        self.list_feature = list_feature

        self.sr = sr
        self.ss = ss
        self.dist = dist

        self.n_fft = n_fft
        self.window = torch.hann_window(self.n_fft)

        self.C= n_channel
        self.F= n_fft//2+1

        channel_feature = 0

        if "X" in list_feature :
            channel_feature += n_channel*2
        if "SV" in list_feature :
            channel_feature += n_channel*2

        self.channel_feature = channel_feature

    def spec(self,X,angle,mic_pos):
        return torch.cat([X.real, X.imag],dim=1)

    def SV(self,X,angle,mic_pos) :
        """
        X : [B*C,F,T]
        angle : [B, 1]
        """
        B, _ = angle.shape
        _, F,T = X.shape
        # init
        B = angle.shape[0]
        loc_src = torch.zeros(B,3).to(mic_pos.device)

        loc_src[:,0] = self.dist*torch.cos((-angle)/180*torch.pi)*torch.sin(torch.tensor(90/180*torch.pi))
        loc_src[:,1] = self.dist*torch.sin((-angle)/180*torch.pi)*torch.sin(torch.tensor(90/180*torch.pi))
        loc_src[:,2] = self.dist*torch.cos(torch.tensor((90)/180*torch.pi))

        TDOA = torch.zeros(B,self.C).to(mic_pos.device)
        for i in range(self.C) : 
            TDOA[:,i] = torch.norm(mic_pos[:,i] - loc_src)
        TDOA = TDOA[:,:] - TDOA[:,0:1]

        const = -1j*2*torch.pi*self.sr/(self.n_fft*self.ss)
        Steering  = torch.zeros(B,self.C,self.F,dtype=torch.cfloat).to(mic_pos.device)
        for i in torch.arange(B) : 
            for j in range(self.F) : 
                Steering[i,:,j] = torch.exp(j*TDOA[i]*const)
                Steering[i,:,j] /= torch.norm(Steering[i,:,j])

        Steering = Steering.unsqueeze(-1).expand(-1, -1, -1, T)
        return torch.cat([Steering.real, Steering.imag],dim=1)

    def forward(self,x,angle,mic_pos):
        B,C,L = x.shape
        # reshape : [B, C, L] -> [B * C, L]
        x = torch.reshape(x,(B*C,L))
        X = torch.stft(x,n_fft = self.n_fft,window=self.window.to(x.device),return_complex=True)
        _, F,T = X.shape
        X = X.reshape(B,C,F,T)

        feature = None
        for type in self.list_feature :
            func = getattr(self, type)
            feat = func(X,angle,mic_pos)
            if feature is None :
                feature = feat
            else :
                feature = torch.cat([feature,feat],dim=1)

        feature = feature.reshape(B,C*self.channel_feature*F,T)

        return feature.permute(0,2,1)



