import torch
import torch.nn as nn

"""
Direction Attractor Net

NOTE : Because there are too many untold components, major parts of this code are improvisations.

Y. Nakagome, M. Togami, T. Ogawa and T. Kobayashi, "Deep Speech Extraction with Time-Varying Spatial Filtering Guided By Desired Direction Attractor," ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, Spain, 2020, pp. 671-675, doi: 10.1109/ICASSP40776.2020.9053629.
"""

class encoder(nn.Module):
    def __init__(self,
                 n_in = 257,
                 n_dim=1024,
                 n_hidden_layer=2,
                 n_out = 40,
                 type_activation=None,
                 type_normalization=None
                 ):
        super(encoder, self).__init__()
        n_dim = 1024

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
            nn.Linear(n_in,n_dim),
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
            nn.Linear(n_dim,n_out),
            activation(),
            normalization(n_out)
            )
        self.add_module("layer_{}".format(n_hidden_layer+1),module)
        self.layers.append(module)


    def forward(self,x):
        for i,layer in enumerate(self.layers):
            x = layer(x)
        return x
    
class estimator(nn.Module):
    def __init__(self,
                 n_in,
                 n_out,
                 type_activation=None,
                 type_normalization=None
                 ) :
        super(estimator,self).__init__()

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

        self.layer = nn.Sequential(
            nn.Linear(n_in,n_out),
            activation(),
            normalization(n_out)
        )
    
    def forward(self,x):
        return self.layer(x)

class DirectionAttractor(nn.Module) : 
    def __init__(self,
                 n_channel=4,
                 n_fft=512,
                 n_dim=1024,
                 n_hidden_layer=2,
                 n_latent = 40,
                 type_activation=None,
                 type_normalization=None,
                 type_activation_out=None
                 ):
        super(DirectionAttractor, self).__init__()
        n_hfft = n_fft//2+1
        n_dim = 1024
        n_dim_out = 1*n_hfft

        self.D = encoder(
                n_in = n_channel*2*n_hfft,
                n_dim=n_dim,
                n_hidden_layer=n_hidden_layer,
                n_out = n_latent,
                type_activation=type_activation,
                type_normalization=type_normalization
        )  
        self.Z_s = encoder(
                n_in = 1,
                n_dim=n_dim,
                n_hidden_layer=n_hidden_layer,
                n_out= n_latent,
                type_activation=type_activation,
                type_normalization=type_normalization
        )
        self.Z_n = encoder(
                n_in = 1,
                n_dim=n_dim,
                n_hidden_layer=n_hidden_layer,
                n_out = n_latent,
                type_activation=type_activation,
                type_normalization=type_normalization
        )

        self.estimation_target_mask = estimator(n_latent,n_dim_out)
        self.estimation_target_activity= estimator(n_latent,n_dim_out)
        self.estimation_noise_mask = estimator(n_latent,n_dim_out)
        self.estimation_noise_activity = estimator(n_latent,n_dim_out)

        if type_activation_out == "Sigmoid" : 
            self.activation = nn.Sigmoid()
        elif type_activation_out == "Tanh" : 
            self.activation = nn.Tanh()
        elif type_activation_out == "ReLU" : 
            self.activation = nn.ReLU()
        else :
            self.activation = nn.Identity()

    def forward(self,spectral_feature,angle):

        e = self.D(spectral_feature)

        a_s = self.Z_s(angle)
        a_n = self.Z_n(angle)

        """
        e : [B, T, F']
        a_s : [B,  F']
        a_n : [B,  F']
        """

        #print("e : {} | a_s {} | a_n {}".format(e.shape,a_s.shape,a_n.shape))

        a_s = a_s.unsqueeze(1)
        a_n = a_n.unsqueeze(1)

        e_s = e * a_s
        e_n = e * a_n

        #print("e_s {} | e_n {}".format(e_s.shape,e_n.shape))

        M_s = self.estimation_target_mask(e_s)
        v_s = self.estimation_target_activity(e_s)
        M_n = self.estimation_noise_mask(e_n)
        v_n = self.estimation_noise_activity(e_n)

        #print("M_s {} | V_s {} | M_n {} | V_n {}".format(M_s.shape, v_s.shape, M_n.shape,v_n.shape) )

        M_s = self.activation(M_s)
        v_s = self.activation(v_s)
        M_n = self.activation(M_n)
        v_n = self.activation(v_n)

        return M_s,v_s,M_n,v_n

class DirectionAttractorNet(nn.Module):
    def __init__(self,
                 n_channel=4,
                 n_fft=512,
                 dist=1,
                 method_out = "Masking",
                 type_activation=None,
                 type_normalization=None,
                 type_activation_out=None
                 ):
        super(DirectionAttractorNet, self).__init__()

        self.sr = 16000
        self.ss = 340.4

        self.n_channel = n_channel
        self.n_fft = n_fft
        self.n_hfft = n_fft//2+1
        self.window = torch.hann_window(self.n_fft)

        self.dist = dist

        self.DAN = DirectionAttractor(
            n_channel=n_channel,
            n_fft=n_fft,
            type_activation=type_activation,
            type_normalization=type_normalization,
            type_activation_out=type_activation_out
            )
        
        if method_out == "Masking" :
            self.f_out = self.masking
        else : 
            raise Exception("{} is not implemented".format(method_out))
        
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
                # TODO : steering vector normalization


        return SV

    def anlge_pre(self,angle):
        return torch.sin((-angle)/180*torch.pi)
    
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
    
    def MVDR(self,X,h,R_s,R_n):
        raise Exception("MVDR()::Not implemented yet")

    """
    X : [B,C,F,T]
    """
    def masking(self,X,h,M_s,v_s,M_n,v_n):
        # W : [B, T, F]
        w = (M_s*v_s)/(M_n*v_n+1e-6)

        w = w.permute(0,2,1)
        Y = w*X[:,0,:,:]
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

        spectral_feauture = torch.abs(torch.cat([X,a],dim=1))
        spectral_feauture = spectral_feauture.reshape(B,C*2*F,T)

        angle  = self.anlge_pre(angle.unsqueeze(-1))


        # [B,C,F,T] -> [B,C,T,F]
        spectral_feauture = spectral_feauture.permute(0,2,1)


        #print("spectral_feature : {}, angle {}".format(spectral_feauture.shape,angle.shape))

        M_s,v_s,M_n,v_n = self.DAN(spectral_feauture,angle)


        Y = self.f_out(X,a,M_s,v_s,M_n,v_n)

        # into batchs
        y = torch.istft(Y,n_fft = self.n_fft,window=self.window.to(x.device))
        # reshape
        return y