import torch
import torch.nn as nn

import pdb

"""
Direction Attractor Net

Inspired by 
Y. Nakagome, M. Togami, T. Ogawa and T. Kobayashi, "Deep Speech Extraction with Time-Varying Spatial Filtering Guided By Desired Direction Attractor," ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 671-675, doi: 10.1109/ICASSP40776.2020.9053629.

NOTE : Because there are too many untold components, major parts of this code are improvisations. 
"""

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

class DirectionAttractor(nn.Module) : 
    def __init__(self,
                 n_channel=4,
                 n_angle=2,
                 n_fft=512,
                 n_dim=1024,
                 n_hidden_layer=4,
                 n_latent = 257,
                 n_rnn_layer=2,
                 type_activation=None,
                 type_normalization=None,
                 type_activation_out=None,
                 dropout = 0.0,
                 out_cplx = False
                 ):
        super(DirectionAttractor, self).__init__()
        n_hfft = n_fft//2+1
        n_dim = 1024
        if out_cplx :
            n_dim_out = 2*n_channel*n_hfft
        else :
            n_dim_out = 1*n_channel*n_hfft

        self.F = n_hfft
        self.C = n_channel

        self.D = encoder(
                n_in = n_channel*4*n_hfft,
                n_dim=n_dim,
                n_hidden_layer=n_hidden_layer,
                n_out = n_latent,
                type_activation=type_activation,
                type_normalization=type_normalization,
                dropout=dropout
        )  
        self.Z_s = encoder(
                n_in = n_angle,
                n_dim=n_dim,
                n_hidden_layer=n_hidden_layer,
                n_out= n_latent,
                type_activation=type_activation,
                type_normalization=type_normalization,
                dropout=dropout
        )
        self.Z_n = encoder(
                n_in = n_angle,
                n_dim=n_dim,
                n_hidden_layer=n_hidden_layer,
                n_out = n_latent,
                type_activation=type_activation,
                type_normalization=type_normalization,
                dropout=dropout
        )

        self.RNN_s = RNN(n_latent,n_latent,n_rnn_layer=n_rnn_layer)
        self.RNN_n = RNN(n_latent,n_latent,n_rnn_layer=n_rnn_layer)

        self.estimation_target_mask = estimator(n_latent,n_dim_out)
        self.estimation_target_activity= estimator(n_latent,1)
        self.estimation_noise_mask = estimator(n_latent,n_dim_out)
        self.estimation_noise_activity = estimator(n_latent,1)

        self.activation_1 = nn.Sigmoid()
        self.activation_2 = nn.Tanh()

    def forward(self,spectral_feature,angle):
        B = spectral_feature.shape[0]
        T = spectral_feature.shape[1]
        e = self.D(spectral_feature)


        angle = torch.unsqueeze(angle,1)
        a_s = self.Z_s(angle)
        a_n = self.Z_n(angle)

        """
        e : [B, T, F']
        a_s : [B,  F']
        a_n : [B,  F']
        """

        #print("e : {} | a_s {} | a_n {}".format(e.shape,a_s.shape,a_n.shape))
        e_s = e * a_s
        e_n = e * a_n

        e_s = self.RNN_s(e_s)
        e_n = self.RNN_n(e_n)

        #print("e_s {} | e_n {}".format(e_s.shape,e_n.shape))

        M_s = self.estimation_target_mask(e_s)
        v_s = self.estimation_target_activity(e_s)
        M_n = self.estimation_noise_mask(e_n)
        v_n = self.estimation_noise_activity(e_n)

        #print("M_s {} | V_s {} | M_n {} | V_n {}".format(M_s.shape, v_s.shape, M_n.shape,v_n.shape) )

        M_s = self.activation_2(M_s)
        v_s = self.activation_1(v_s)
        M_n = self.activation_2(M_n)
        v_n = self.activation_1(v_n)

        M_s = torch.reshape(M_s,(B,T,self.F,self.C))
        M_n = torch.reshape(M_n,(B,T,self.F,self.C))
        """
        M_s = torch.reshape(M_s,(B,T,self.F,self.C,2))
        M_n = torch.reshape(M_n,(B,T,self.F,self.C,2))
        M_s = M_s[...,0] + 1j*M_s[...,1]
        M_n = M_n[...,0] + 1j*M_n[...,1]
        """

        return M_s,v_s,M_n,v_n

class DirectionAttractorNet(nn.Module):
    def __init__(self,
                 n_channel=4,
                 n_fft=512,
                 dist=1,
                 method_out = "mask_mag",
                 type_activation=None,
                 type_normalization=None,
                 type_activation_out=None,
                 dropout = 0.0,
                 angle_feature = "theta"
                 ):
        super(DirectionAttractorNet, self).__init__()

        self.sr = 16000
        self.ss = 340.4

        self.n_channel = n_channel
        self.n_fft = n_fft
        self.n_hfft = n_fft//2+1
        self.window = torch.hann_window(self.n_fft)

        self.dist = dist
        
        if method_out == "mask_mag" :
            self.f_out = self.masking
            self.out_cplx = False
        elif method_out == "mask_cplx" : 
            self.f_out = self.masking
            self.out_cplx = True
        elif method_out == "MVDR" :
            self.f_out = self.MVDR
        else : 
            raise Exception("{} is not implemented".format(method_out))
        
        self.anlgle_feature = angle_feature
        if angle_feature == "theta" : 
            n_angle = 2
        elif angle_feature == "absSV" : 
            n_angle = self.n_hfft*n_channel
            raise Exception("{} is not implemented".format(angle_feature))
        elif angle_feature == "SV" : 
            n_angle = self.n_hfft*2*n_channel
            raise Exception("{} is not implemented".format(angle_feature))
        else :
            raise Exception("{} is unknown type of anlge feature".format(angle_feature))
        
        self.DAN = DirectionAttractor(
            n_channel=n_channel,
            n_fft=n_fft,
            type_activation=type_activation,
            type_normalization=type_normalization,
            type_activation_out=type_activation_out,
            dropout=dropout,
            n_angle = n_angle,
            out_cplx = self.out_cplx
            )
    
        
    def steering_vector(self,angle,mic_pos) :
        """
        angle : [B, 1]
        """
        # init
        B = angle.shape[0]
        loc_src = torch.zeros(B,3).to(mic_pos.device)

        loc_src[:,0] = self.dist*torch.cos((-angle)/180*torch.pi)*torch.sin(torch.tensor(90/180*torch.pi))
        loc_src[:,1] = self.dist*torch.sin((-angle)/180*torch.pi)*torch.sin(torch.tensor(90/180*torch.pi))
        loc_src[:,2] = self.dist*torch.cos(torch.tensor((90)/180*torch.pi))

        TDOA = torch.zeros(B,self.n_channel).to(mic_pos.device)
        for i in range(self.n_channel) : 
            TDOA[:,i] = torch.norm(mic_pos[:,i] - loc_src)
        TDOA = TDOA[:,:] - TDOA[:,0:1]

        const = -1j*2*torch.pi*self.sr/(self.n_fft*self.ss)
        SV  = torch.zeros(B,self.n_channel,self.n_hfft,dtype=torch.cfloat).to(mic_pos.device)
        for i in torch.arange(B) : 
            for j in range(self.n_hfft) : 
                SV[i,:,j] = torch.exp(j*TDOA[i]*const)
                SV[i,:,j] /= torch.norm(SV[i,:,j])

        return SV

    def anlge_pre(self,angle):
        sin_theta = torch.sin((-angle)/180*torch.pi)
        cos_theta = torch.cos((-angle)/180*torch.pi)
        return torch.stack((sin_theta,cos_theta),1)
    
    def cal_covariance(self,X,M_s,v_s,M_n,v_n) : 
        raise Exception("cal_covariance()::Not implemented yet")
        # X : [B, C, F, T]
        B,C,F,T = X.shape

        X_ = torch.permute(X,(0,2,3,1))
        X_ = torch.reshape(X_,(B*F*T,C,1))

        R = torch.bmm(X_,torch.permute(X_,(0,2,1)))
        # diagonal loading
        R += torch.eye(C)*1e-4

        R_inv = torch.inverse(R)

        W = v_s*M_s*X

        Y = W*X
        return Y
    """
    X = B,C,F,T
    h = B,C,F,T
    M_n = B,T,F,C
    v_n = B,T,1
    """
    def MVDR(self,X,h,M_s,v_s,M_n,v_n):
        B,C,F,T = X.shape

        v_n = torch.reshape(v_n,(B,T,1,1))
        G_n = M_n*v_n

        # B,T,F,C -> B,F,T,C
        G_n = torch.permute(G_n,(0,2,1,3))
        G_n = torch.reshape(G_n,(B*F*T,1,C))

        X = torch.permute(X,(0,2,3,1))
        X = torch.reshape(X,(B*F*T,1,C))

        GnX = 0.5*G_n*X + 0.5*X

        Rn = torch.transpose(GnX,1,2)*GnX

        h = torch.permute(h,(0,2,3,1))
        h = torch.reshape(h,(B*F*T,C,1))

        hT = torch.transpose(h,1,2)

        # invsese 
        Rn_inv = torch.inverse(Rn + torch.eye(C).to(Rn.device)*1e-4)

        numer = torch.bmm(Rn_inv,h)
        denom = torch.bmm(torch.bmm(hT,Rn_inv),h)

        w = numer/denom

        bY = torch.bmm(X,w)

        Y = torch.reshape(bY,(B,F,T))
        return Y

    """
    X : [B,C,F,T]
    """
    def masking(self,X,h,M_s,v_s,M_n,v_n):
        # TODO : complex mask

        B,T,F,C = M_s.shape
        v_s = torch.reshape(v_s,(B,T,1,1))
        v_n = torch.reshape(v_s,(B,T,1,1))

        # W : [B, T, F]
        #w = (M_s*v_s)/(M_n*v_n+1e-6)

        w = M_s*v_s - M_n*v_n
        #w = (M_s*v_s)

        w = w.permute(0,2,1,3)
        Y = w[:,:,:,0]*X[:,0,:,:]
        return Y

    """
    x : [B, C, L], mutli-channel time-domain signal
    y : [B, 1, L], estimated time-domain signal
    """
    def forward(self,x,angle,mic_pos):
        # reshape : [B, C, L] -> [B * C, L]
        B,C,L = x.shape
        x = torch.reshape(x,(B*C,L))
        X = torch.stft(x,n_fft = self.n_fft,window=self.window.to(x.device),return_complex=True)
        _, F,T = X.shape
        X = X.reshape(B,C,F,T)

        # reshape :
        a = self.steering_vector(angle,mic_pos)
        a = a.unsqueeze(-1).expand(-1, -1, -1, T)

        spectral_feature = torch.cat([X.real,X.imag,a.real,a.imag],dim=1)
        spectral_feature = spectral_feature.reshape(B,C*4*F,T)

 

        # [B,C,F,T] -> [B,C,T,F]
        spectral_feature = spectral_feature.permute(0,2,1)
        
        #print("spectral_feature : {}, angle {}".format(spectral_feature.shape,angle.shape))

        if self.anlgle_feature == "theta" : 
            angle_feature = self.anlge_pre(angle)
        elif self.anlgle_feature =="SV" :
            angle_feature = torch.cat((a.real,a.imag),dim=1)
        elif self.anlgle_feature =="absSV" :
            angle_feature = torch.abs(torch.cat((a.real,a.imag),dim=1))

        M_s,v_s,M_n,v_n = self.DAN(spectral_feature,angle_feature)

        Y = self.f_out(X,a,M_s,v_s,M_n,v_n)

        # into batchs
        y = torch.istft(Y,n_fft = self.n_fft,window=self.window.to(x.device))
        # reshape
        return y