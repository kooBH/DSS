"""
UNet based Directional Source Separation
"""

import torch
import torch.nn as nn

try : 
    from .UNet_m import ComplexConv2d, ComplexConvTranspose2d, ComplexBatchNorm2d, ComplexDepthSeparable, MEA
except ImportError:
    from UNet_m import ComplexConv2d, ComplexConvTranspose2d, ComplexBatchNorm2d, ComplexDepthSeparable, MEA

class FGRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels):
        super(FGRUBlock, self).__init__()
        self.GRU = nn.GRU(
            in_channels*2, hidden_size*2, batch_first=True, bidirectional=True
        )
        # the GRU is bidirectional -> multiply hidden_size by 2
        self.conv = ComplexConv2d(hidden_size * 2, out_channels, kernel_size=1)
        self.bn = ComplexBatchNorm2d(out_channels)
        self.hidden_size = hidden_size
        self.relu = nn.PReLU()

    # x : torch.Size([B, C=128, F=2, T=16, RI=2])
    def forward(self, x):
        B, C, F, T, _ = x.shape
        x_ = x.permute(0, 3, 2, 1,4)  # x_.shape == (B,T,F,C,2)
        x_ = x_.reshape(B * T, F, C*2)
        y, h = self.GRU(x_)  # x_.shape == (BT,F,C*2)
        y = y.reshape(B, T, F, self.hidden_size*2,2)
        output = y.permute(0, 3, 2, 1, 4)  # output.shape == (B,C,F,T,2)
        output = self.conv(output)
        output = self.bn(output)
        return self.relu(output)

class TGRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, skipGRU=False,**kwargs):
        super(TGRUBlock, self).__init__()

        if not skipGRU : 
            self.GRU = nn.GRU(in_channels*2, hidden_size*2, batch_first=True)
        else : 
            raise Exception("Not Implemented")
            #self.GRU = SkipGRU(in_channels*2, hidden_size*2, batch_first=True)
        self.conv = ComplexConv2d(hidden_size, out_channels, kernel_size=1)
        self.bn = ComplexBatchNorm2d(out_channels)
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()

    # x : torch.Size([B, C=128, F=2, T=16, RI=2])
    def forward(self, x, rnn_state=None):
        B, C, F, T, _ = x.shape  # x.shape == (B, C, T, F)

        # unpack, permute, and repack
        x1 = x.permute(0, 2, 3, 1, 4)  # x2.shape == (B,F,T,C,2)
        x_ = x1.reshape(B * F, T, C*2)  # x_.shape == (BF,T,C*2)

        # run GRU
        y_, rnn_state = self.GRU(x_, rnn_state)  # y_.shape == (BF,T,C*2)
        # unpack, permute, and repack
        y1 = y_.reshape(B, F, T, self.hidden_size,2)  # y1.shape == (B,F,T,C,2)
        y2 = y1.permute(0, 3, 1, 2, 4)  # y2.shape == (B,C,F,T,2)

        output = self.conv(y2)
        output = self.bn(output)
        output = self.relu(output)
        return output
    
class FTGRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels,**kwargs):
        super(FTGRUBlock, self).__init__()
        
        self.FGRU = FGRUBlock(in_channels, hidden_size, out_channels,**kwargs)
        self.TGRU = TGRUBlock(in_channels, hidden_size, out_channels,**kwargs)
    
    def forward(self, x, rnn_state = None) : 
        x = self.FGRU(x)
        x = self.TGRU(x,rnn_state)
    
        return x

class permuteTF(nn.Module):
    def __init__(self):
        super(permuteTF, self).__init__()

    def forward(self,x):
        if len(x.shape) == 3 :
            x = torch.permute(x,(0,2,1))
        elif len(x.shape) == 4 :
            x = torch.permute(x,(0,1,3,2))
        return x
    
class cRNN(nn.Module) : 
    def __init__(self,dim,
                 style="GRU"
                 ):
        super(cRNN, self).__init__()
    
        if style == "GRU" : 
            self.rr = nn.GRU(dim,dim,batch_first=True)
            self.ri = nn.GRU(dim,dim,batch_first=True)
            self.ir = nn.GRU(dim,dim,batch_first=True)
            self.ii = nn.GRU(dim,dim,batch_first=True)
        elif style == "LSTM" : 
            self.rr = nn.LSTM(dim,dim,batch_first=True)
            self.ri = nn.LSTM(dim,dim,batch_first=True)
            self.ir = nn.LSTM(dim,dim,batch_first=True)
            self.ii = nn.LSTM(dim,dim,batch_first=True)
    
        self.re_norm = nn.BatchNorm1d(dim)
        self.im_norm = nn.BatchNorm1d(dim)
        self.re_activation = nn.PReLU()
        self.im_activation = nn.PReLU()
        
    def forward(self,x):
        # x : [B,C,F,T,2] -> [2,B,T,C*F]
        B,C,F,T,_ = x.shape
        x = torch.permute(x,(4,0,3,1,2))
        x = torch.reshape(x,(2,B,T,C*F))
        
        rr = self.rr(x[0])[0]
        ri = self.ri(x[0])[0]
        ir = self.ir(x[1])[0]
        ii = self.ii(x[1])[0]
        
        re = rr - ii
        im = ri + ir
        
        # re : [B,T,C*F]
        # im = [B,T,C*F]
        
        re = self.re_norm(torch.permute(re,(0,2,1)))
        im = self.im_norm(torch.permute(im,(0,2,1)))
        
        re = self.re_activation(re)
        im = self.im_activation(im)
        
        # x  : [2,B,T,C*F] -> [B,C*F,T,2] -> [B,C,F,T,2]
        x = torch.stack((re,im),dim=-1)
        x = torch.permute(x,(1,3,2,0))
        x = torch.reshape(x,(B,C,F,T,2))
        return x


class Attractor(nn.Module):
    def __init__(self,
                 n_in = 257,
                 n_dim=1024,
                 n_hidden_layer=1,
                 n_out = 40,
                 type_activation=None,
                 type_normalization=None,
                 dropout = 0.0
                 ):
        super(Attractor, self).__init__()
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

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, complex=True, padding_mode="zeros",activation="LeakyReLU"):
        super().__init__()
        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # 'SAME' padding
            
        if complex:
            conv = ComplexConv2d
            bn = ComplexBatchNorm2d
        else:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d

        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
        self.bn = bn(out_channels)

        if activation == "LeakyReLU" : 
            self.relu = nn.LeakyReLU(inplace=True)
        elif activation == "PReLU" :
            self.relu = nn.PReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding,padding=(0, 0), complex=True,activation = "LeakyReLU"):
        super().__init__()
        if complex:
            tconv = ComplexConvTranspose2d
            bn = ComplexBatchNorm2d
        else:
            tconv = nn.ConvTranspose2d
            bn = nn.BatchNorm2d
        
        self.transconv = tconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=output_padding,padding=padding)
        self.bn = bn(out_channels)

        if activation == "LeakyReLU" : 
            self.relu = nn.LeakyReLU(inplace=True)
        elif activation == "PReLU" :
            self.relu = nn.PReLU(inplace=True)

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class UDSS(nn.Module):
    def __init__(self, 
                 input_channels=4,
                 n_fft=512,
                 complex=True,
                 #model_complexity=45,
                 model_complexity=45,
                 bottleneck="None",
                 padding_mode="zeros",
                 type_encoder = "Complex",
                 type_masking = "CRM",
                 dropout=0.0):
        super().__init__()

        self.complex = complex
        self.n_channel = input_channels
        n_angle = 2

        if not complex:
            input_channels *=2
        else  :
            model_complexity = int(model_complexity // 1.414)

        print("UDSS::complexity {}".format(model_complexity))

        model_depth=20

        self.set_size(model_complexity=model_complexity, input_channels=input_channels, model_depth=model_depth)
        self.model_length = model_depth // 2
        self.dropout = dropout

        ## Encoder
        self.encoders = []

        if type_encoder == "Complex" : 
            module_cls = Encoder
        elif type_encoder == "ComplexDepthSeparable" : 
            module_cls = ComplexDepthSeparable
        else : 
            raise Exception("Not Implemented")

        for i in range(self.model_length):
            module = module_cls(self.enc_channels[i], self.enc_channels[i + 1], kernel_size=self.enc_kernel_sizes[i],
                             stride=self.enc_strides[i], padding=self.enc_paddings[i], padding_mode=padding_mode)
            self.add_module("encoder{}".format(i), module)
            self.encoders.append(module)

        ## Decoder
        self.decoders = []

        for i in range(self.model_length):
            module = Decoder(self.dec_channels[i] + self.enc_channels[self.model_length - i], self.dec_channels[i + 1], kernel_size=self.dec_kernel_sizes[i],
                             stride=self.dec_strides[i], padding=self.dec_paddings[i], output_padding=self.dec_output_paddings[i])
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)

        # Bottleneck
        if bottleneck == "cRNN" : 
            self.RNN = cRNN(128*2)
        elif bottleneck == "TGRU":
            self.RNN = TGRUBlock(128,256,128)
        elif bottleneck == "FTGRU" : 
            self.RNN = FTGRUBlock(128,256,128)
        else :
            self.RNN = nn.Identity()
            
        ## Attractor
        self.attractors =  [] 

        for i in range(self.model_length-1) : 
            module = Attractor(
                n_in=n_angle,
                n_dim=257,
                n_out = self.enc_channels[i+1]
            )
            self.add_module("attractor{}".format(i),module)
            self.attractors.append(module)
        self.bottleneck_attactor = Attractor(
            n_in = n_angle,
            n_dim =257,
            n_out = self.enc_channels[-1]
        )

        if complex:
            conv = ComplexConv2d
            linear = conv(self.dec_channels[-1], 1, 1)
        else:
            conv = nn.Conv2d
            linear = conv(self.dec_channels[-1], 2, 1)

        ## Mask Estimator
        self.add_module("linear", linear)
        self.complex = complex
        self.padding_mode = padding_mode

        if type_masking == "CRM" : 
            self.masking = nn.Identity()
        elif type_masking == "MEA" : 
            self.masking = MEA()
        else :
            raise Exception("Not Implemented")

        self.dr = nn.Dropout(self.dropout)

        self.decoders = nn.ModuleList(self.decoders)
        self.encoders = nn.ModuleList(self.encoders)
        self.attractors = nn.ModuleList(self.attractors)

    def forward(self, sf,af):        
        # ipnut : [ Batch Channel Freq Time 2]

        # Encoders
        sf_skip = []
        for i, encoder in enumerate(self.encoders):
            sf_skip.append(sf)
            sf = encoder(sf)
            sf = self.dr(sf)
            #print("sf{}".format(i), sf.shape)
        # sf_skip : sf0=input sf1 ... sf9

        #print("fully encoded ",sf.shape)
        p = sf

        p = self.RNN(p)

        # Bottleneck
        a_s = self.bottleneck_attactor(af)
        a_s = torch.reshape(a_s,(*a_s.shape,1,1,1))
        p = a_s*p

        # Attractor - Skip
        for i,attractor in enumerate(self.attractors) : 
            a_s = attractor(af)
            a_s = torch.reshape(a_s,(*a_s.shape,1,1,1))
            sf_skip[i+1] = a_s*sf_skip[i+1]
            #print("attractor[{}] : {}*{}".format(i,a_s.shape,sf_skip[i+1].shape))
        
        # Decoders
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            if i == self.model_length - 1:
                break
            #print(f"p{i}, {p.shape} + sf{self.model_length - 1 - i}, {sf_skip[self.model_length - 1 -i].shape}, padding {self.dec_paddings[i]}")
            
            p = torch.cat([p, sf_skip[self.model_length - 1 - i]], dim=1)

        #:print(p.shape)
        mask = self.linear(p)
        mask = torch.tanh(mask)
        mask = torch.squeeze(mask,1)
        mask = mask[...,0] + 1j*mask[...,1]

        return mask

    def set_size(self, model_complexity, model_depth=20, input_channels=1):
        self.enc_channels = [input_channels,
                                model_complexity,
                                model_complexity,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                128]

        self.enc_kernel_sizes = [(7, 1),
                                    (1, 7),
                                    (7, 5),
                                    (7, 5),
                                    (5, 3),
                                    (5, 3),
                                    (5, 3),
                                    (5, 3),
                                    (5, 3),
                                    (5, 3)]

        self.enc_strides = [(1, 1),
                            (1, 1),
                            (2, 2),
                            (2, 1),
                            (2, 2),
                            (2, 1),
                            (2, 2),
                            (2, 1),
                            (2, 2),
                            (2, 1)]

        self.enc_paddings = [(3, 0),
                                (0, 3),
                                (3, 2),
                                (3, 2),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),]
                            
                                

        self.dec_channels = [0,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2,
                                model_complexity * 2]

        self.dec_kernel_sizes = [(5, 3),
                                    (5, 3),
                                    (5, 3),
                                    (5, 3),
                                    (5, 3),
                                    (5, 3), 
                                    (7, 5), 
                                    (7, 5), 
                                    (1, 7),
                                    (7, 1)]

        self.dec_strides = [(2, 1),
                            (2, 2),
                            (2, 1),
                            (2, 2),
                            (2, 1),
                            (2, 2),
                            (2, 1),
                            (2, 2),
                            (1, 1),
                            (1, 1)]

        self.dec_paddings = [(2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (2, 1),
                                (3, 2),
                                (3, 2),
                                (0, 3),
                                (3, 0)]
        self.dec_output_paddings = [(0,0),
                                    (0,1),
                                    (0,0),
                                    (0,1),
                                    (0,0),
                                    (0,1),
                                    (0,0),
                                    (0,1),
                                    (0,0),
                                    (0,0)]
        
class UDSS_helper(nn.Module):
    def __init__(self,
                 n_fft = 512,
                 complex = True,
                 dropout = 0.0,
                 bottleneck = "None",
                 model_complexity = 45,
                 type_encoder = "Complex",
                 use_SV = True,
                 mag_phase = False
                 ):
        super(UDSS_helper,self).__init__()

        self.n_fft = n_fft
        self.n_hfft = n_fft // 2 + 1

        self.n_channel = 4

        self.complex  = complex
        self.mag_phase = mag_phase

        self.use_SV = use_SV


        if self.use_SV : 
            in_channel = self.n_channel*2
        else  : 
            in_channel = self.n_channel

        self.net = UDSS(input_channels = in_channel,
                        complex = complex,
                        dropout = dropout,
                        bottleneck=bottleneck,
                        model_complexity=model_complexity,
                        type_encoder=type_encoder
                        )

        # const
        self.sr = 16000
        self.ss = 340.4
        self.dist = 100

        self.window = torch.hann_window(self.n_fft)

    def forward(self,x,angle,mic_pos):
        B,C,L = x.shape

        x = torch.reshape(x,(B*C,L))
        X = torch.stft(x,n_fft = self.n_fft,window=self.window.to(x.device),return_complex=True)
        _, F,T = X.shape
        short = 16-T%16
        X = torch.nn.functional.pad(X,(0,short))
        _, F,T = X.shape
        X = X.reshape(B,C,F,T)

        if self.use_SV : 
            SV = self.steering_vector(angle,mic_pos)
            SV = SV.unsqueeze(-1).expand(-1, -1, -1, T)

        angle_feature = self.anlge_pre(angle)
        #print("angle feaute : {}".format(angle_feature.shape))

        if self.complex : 
            # [B,C,F,T,2]
            spectral_feature = torch.stack((X.real,X.imag),dim=-1)

            if self.use_SV : 
                spectral_feature = torch.cat((spectral_feature,torch.stack((SV.real,SV.imag),dim=-1)),dim=1)
        else : 
            if not self.mag_phase : 
                spectral_feature = torch.cat((X.real,X.imag),dim=1)
            else :
                import pdb
                pdb.set_trace()
                spectral_feature = torch.cat((X.real,X.imag),dim=1)
        

        mask = self.net(spectral_feature,angle_feature)

        Y = X[:,0]*mask
        y = torch.istft(Y,n_fft = self.n_fft,window=self.window.to(Y.device),length=L)
        return y

    def anlge_pre(self,angle):
        sin_theta = torch.sin((-angle)/180*torch.pi)
        cos_theta = torch.cos((-angle)/180*torch.pi)
        return torch.stack((sin_theta,cos_theta),1)
    
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

if __name__ == "__main__" : 
    B = 2
    C = 4
    F = 257
    L = 32000
    T = 256

    #x = torch.rand(B,C,128*127)
    sf = torch.rand(B,C*2,F,T,2)
    af = torch.rand(B,2)
    m = UDSS(input_channels=C*2)

    y = m(sf,af)

    print("output : {}".format(y.shape))