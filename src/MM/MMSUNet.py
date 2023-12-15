"""
Directional Separating UNet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
try : 
    from .lipfeat import LipNet
except ImportError:
    from lipfeat import LipNet


class Attractor(nn.Module) :
    def __init__(self,n_ch=4,n_fft=512):
        super(Attractor,self).__init__()

        self.enc_SV = nn.Linear(n_ch *(n_fft+2), n_fft//2 +1)
        self.act_SV = nn.PReLU()

        self.enc_theta = nn.Linear(2,n_fft//2 +1)
        self.act_theta = nn.Sigmoid()

        self.bn = nn.BatchNorm1d(n_fft//2 +1)

        self.enc_2 = nn.Linear(n_fft//2 +1, n_fft+2)
        self.act_2 = nn.PReLU()
        self.enc_3 = nn.Linear(n_fft+2, n_fft//2)
        self.act_3 = nn.Sigmoid()

    def forward(self, SV, theta):
        SV = torch.reshape(SV,(SV.shape[0],-1))
        a = self.enc_SV(SV)
        a = self.act_SV(a)

        b = self.enc_theta(theta)
        b = self.act_theta(b)

        attract = a + b
        attract = self.bn(attract)
        attract = self.enc_2(attract)
        attract = self.act_2(attract)
        attract = self.enc_3(attract)
        attract = self.act_3(attract)

        return attract
    
class AttractEncoder(nn.Module):
    def __init__(self,n_ch, n_fft = 512):
        super(AttractEncoder,self).__init__()
        self.n_ch = n_ch

        self.enc= nn.Linear(n_fft//2,n_ch)
        self.conv = nn.Conv1d(1,n_ch,kernel_size=1)
        self.act = nn.Sigmoid()

    # attract : [B,F]
    def forward(self,attract):
        attract = self.enc(attract)
        # attract : [B, n_ch]
        attract = self.act(attract)
        attract = torch.clamp(attract,min=1/self.n_ch)
        return attract

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, padding_mode="zeros",activation="LeakyReLU",dropout=0.0,type_norm = "BatchNorm2d"):
        super().__init__()
        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # 'SAME' padding
       
        conv = nn.Conv2d
        if type_norm == "BatchNorm2d" : 
            bn = nn.BatchNorm2d
        elif type_norm == "InstanceNorm2d" :
            bn = nn.InstanceNorm2d

        self.dr = nn.Dropout(dropout)

        self.conv = conv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
        self.bn = bn(out_channels)

        if activation == "LeakyReLU" : 
            self.relu = nn.LeakyReLU(inplace=True)
        elif activation == "PReLU" :
            self.relu = nn.PReLU()
        elif activation == "SiLU" :
            self.relu = nn.SiLU()
        elif activation == "Softplus" :
            self.relu = nn.Softplus()

    def forward(self, x):
        x = self.conv(x)
        x = self.dr(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    


# Conv Next Style DepthWise Separable Encoder
class EncoderDWN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None, padding_mode="zeros",activation="LeakyReLU",dropout=0.0,type_norm = "BatchNorm2d"):
        super().__init__()
        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # 'SAME' padding
       
        conv = nn.Conv2d
        if type_norm == "BatchNorm2d" : 
            bn = nn.BatchNorm2d
        elif type_norm == "InstanceNorm2d" :
            bn = nn.InstanceNorm2d

        self.dr = nn.Dropout(dropout)

        self.conv_depth = conv(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
        self.conv_point = conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0, padding_mode=padding_mode)
        self.bn = bn(out_channels)

        if activation == "LeakyReLU" : 
            self.relu = nn.LeakyReLU(inplace=True)
        elif activation == "PReLU" :
            self.relu = nn.PReLU()
        elif activation == "SiLU" :
            self.relu = nn.SiLU()
        elif activation == "Softplus" :
            self.relu = nn.Softplus()

    def forward(self, x):
        x = self.conv_depth(x)
        x = self.dr(x)
        x = self.bn(x)
        x = self.conv_point(x)
        x = self.relu(x)
        return x
    
# Gate Convolution Layer
class GConv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0,  groups=1,type_norm = "BatchNorm2d",activation="ReLU",dropout=0.0):
        super(GConv,self).__init__()
        
        self.c = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,  groups=groups)
        self.m = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding,  groups=groups),
            nn.Sigmoid()
            )
        
        if type_norm == "BatchNorm2d" : 
            self.norm= nn.BatchNorm2d(out_channels)
        elif type_norm == "InstanceNorm2d" : 
            self.norm= nn.InstanceNorm2d(out_channels,track_running_stats=True)
        else :
            self.norm= nn.Identity()

        if activation == "ReLU" : 
            self.activation = torch.nn.ReLU()
        elif activation == "SiLU":
            self.activation = torch.nn.SiLU()
        elif activation == "ELU":
            self.activation = torch.nn.ELU()
        elif activation == "Softplus" : 
            self.activation = torch.nn.Softplus()
        elif activation == "PReLU":
            self.activation = torch.nn.PReLU()
        elif activation == "LeakyReLU" :
            self.activation = torch.nn.LeakyReLU()
        
    def forward(self, x): 
        m = self.m(x)
        x = self.c(x)

        y = m*x
        y = self.norm(y)
        y = self.activation(y)
        
        return y

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding,padding=(0, 0), activation = "LeakyReLU",type_norm = "BatchNorm2d"):
        super().__init__()
        
        tconv = nn.ConvTranspose2d
        if type_norm == "BatchNorm2d" : 
            bn = nn.BatchNorm2d
        elif type_norm == "InstanceNorm2d" :
            bn = nn.InstanceNorm2d
        
        self.transconv = tconv(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=output_padding,padding=padding)
        self.bn = bn(out_channels)

        if activation == "LeakyReLU" : 
            self.relu = nn.LeakyReLU(inplace=True)
        elif activation == "PReLU" :
            self.relu = nn.PReLU()
        elif activation == "SiLU" :
            self.relu = nn.SiLU()
        elif activation == "Softplus" :
            self.relu = nn.Softplus()

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class DecoderDWN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding,padding=(0,0),type_norm = "BatchNorm2d",activation = "ReLU"):
        super(DecoderDWN, self).__init__()
        self.out_channels = out_channels
        self.conv_point = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        if type_norm == "BatchNorm2d" : 
            self.norm_point = nn.BatchNorm2d(out_channels)
        elif type_norm == "InstanceNorm2d" : 
            self.norm_point = nn.InstanceNorm2d(out_channels,track_running_stats=True)
        else :
            self.norm_point = nn.Identity()

        self.conv_up = nn.ConvTranspose2d(
            in_channels=out_channels,  # because it passed already in the previous conv
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )

        if activation == "LeakyReLU" : 
            self.activation= nn.LeakyReLU(inplace=True)
        elif activation == "ReLU" : 
            self.activation = torch.nn.ReLU()
        elif activation == "SiLU":
            self.activation = torch.nn.SiLU()
        elif activation == "Softplus" :
            self.relu = nn.Softplus()

    def forward(self, x):
        x = self.conv_point(x)
        x = self.norm_point(x)
        x = self.conv_up(x)
        #x = F.gelu(x)
        x = self.activation(x)
        return x
    
class TrGConv(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1,type_norm = "BatchNorm2d",activation = "ReLU",dropout=0.0):
        super(TrGConv,self).__init__()
        
        self.c = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups)
        self.m = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups),
            nn.Sigmoid()
            )
        
        if type_norm == "BatchNorm2d" : 
            self.norm_point = nn.BatchNorm2d(out_channels)
        elif type_norm == "InstanceNorm2d" : 
            self.norm_point = nn.InstanceNorm2d(out_channels,track_running_stats=True)
        else :
            self.norm_point = nn.Identity()

        if activation == "ReLU" : 
            self.activation = torch.nn.ReLU()
        elif activation == "SiLU":
            self.activation = torch.nn.SiLU()
        elif activation == "Softplus" : 
            self.activation = torch.nn.Softplus()
        elif activation == "ELU":
            self.activation = torch.nn.ELU()
        elif activation == "PReLU":
            self.activation = torch.nn.PReLU()
        elif activation == "LeakyReLU" :
            self.activation = torch.nn.LeakyReLU()
        
    def forward(self, x): 
        m = self.m(x)
        x = self.c(x)
        
        return m*x 


    

    
class TGRUBlock(nn.Module):
    def __init__(self, in_channels, hidden_size, out_channels, skipGRU=False,**kwargs):
        super(TGRUBlock, self).__init__()

        if not skipGRU : 
            self.GRU = nn.GRU(in_channels, hidden_size, batch_first=True)
        else : 
            raise Exception("Not Implemented")
            #self.GRU = SkipGRU(in_channels*2, hidden_size*2, batch_first=True)
        self.conv = nn.Conv2d(hidden_size, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.hidden_size = hidden_size
        self.relu = nn.ReLU()

    # x : torch.Size([B, C=128, F=2, T=16, RI=2])
    def forward(self, x, rnn_state=None):
        B, C, F, T = x.shape  # x.shape == (B, C, F, T)

        # unpack, permute, and repack
        x1 = x.permute(0, 2, 3, 1)  # x2.shape == (B,F,T,C)
        x_ = x1.reshape(B * F, T, C)  # x_.shape == (BF,T,C*2)

        # run GRU
        y_, rnn_state = self.GRU(x_, rnn_state)  # y_.shape == (BF,T,C*2)
        # unpack, permute, and repack
        y1 = y_.reshape(B, F, T, self.hidden_size)  # y1.shape == (B,F,T,C,2)
        y2 = y1.permute(0, 3, 1, 2)  # y2.shape == (B,C,F,T,2)

        output = self.conv(y2)
        output = self.bn(output)
        output = self.relu(output)
        return output
    

################ Reisudal Connection

class SEBlock(nn.Module) :
    def __init__(self,in_channel,ratio=16):
        super().__init__()
        
        size_se = int(in_channel/ratio)
       
        self.pool = torch.nn.AdaptiveAvgPool2d((1,1))
            # Squeeze
        self.squeeze = nn.Linear(in_channel,size_se)
        self.relu = nn.ReLU()
            # Excitation
        self.excitation = nn.Linear(size_se,in_channel)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        # x : [B,C,F,T]
        
        
        x_ = self.pool(x)
        x_ = x_.permute(0,2,3,1)
        
        x_ = self.squeeze(x_)
        x_ = self.relu(x_)
        x_ = self.excitation(x_)
        x_ = self.sigmoid(x_)
        x_ = x_.permute(0,3,1,2)
        x = x*x_
        
       
        return x

################ Output
    
# omplex Ratio Mask
class CRM(nn.Module):
    def __init__(self):
        super(CRM,self).__init__()
    
    def forward(self,x):
        mask = torch.tanh(x)
        mask = mask[:,0,...] + 1j*mask[:,1,...]

        return mask
    
    def output(self,X,M):
        return X*M
    
class CRM_sig(nn.Module):
    def __init__(self):
        super(CRM_sig,self).__init__()
    
    def forward(self,x):
        mask = torch.sigmoid(x)
        mask = mask[:,0,...] + 1j*mask[:,1,...]

        return mask
    
    def output(self,X,M):
        return X*M
    
class CRM_soft(nn.Module):
    def __init__(self):
        super(CRM_soft,self).__init__()

        self.masking = nn.Softplus()
    
    def forward(self,x):
        mask = self.masking(x)
        mask = mask[:,0,...] + 1j*mask[:,1,...]

        return mask
    
    def output(self,X,M):
        return X*M

# Complex Mapping
class CM(nn.Module):
    def __init__(self):
        super(CM,self).__init__()

    def forward(self,x):
        map= x[:,0,...] + 1j*x[:,1,...]
        return map
    
    def output(self,X,M):
        return M


# Complex Mask Estimation and Applying    
class CMEA(nn.Module):
    # class of mask estimation and applying
    def __init__(self,in_channels=1, mag_f_dim=3):
        super(CMEA, self).__init__()
        self.mag_mask = nn.Conv2d(
            in_channels, 1, kernel_size=(3, 1), padding=(1, 0))
        self.real_mask = nn.Conv2d(in_channels, 1, kernel_size=(3, 1), padding=(1, 0))
        self.imag_mask = nn.Conv2d(in_channels, 1, kernel_size=(3, 1), padding=(1, 0))
        self.mag_f_dim = mag_f_dim

    #define custom_atan2 to support onnx conversion
    @staticmethod
    def custom_atan2(y, x):
        pi = torch.from_numpy(np.array([np.pi])).to(y.device, y.dtype)
        ans = torch.atan(y / (x + 1e-6))
        ans += ((y > 0) & (x < 0)) * pi
        ans -= ((y < 0) & (x < 0)) * pi
        ans *= 1 - ((y > 0) & (x == 0)) * 1.0
        ans += ((y > 0) & (x == 0)) * (pi / 2)
        ans *= 1 - ((y < 0) & (x == 0)) * 1.0
        ans += ((y < 0) & (x == 0)) * (-pi / 2)
        return ans
    
    # x : [B,C,F,T,2]
    def forward(self, x):
        mag_mask = self.mag_mask(torch.sqrt(x[...,0]**2+x[...,1]**2)).squeeze(1)
        real_mask = self.real_mask(x[...,0]).squeeze(1)
        imag_mask = self.imag_mask(x[...,1]).squeeze(1)

        return (mag_mask,real_mask,imag_mask)

    def output(self,feature,mask,eps=1e-9) :
        mag_mask  = mask[0]
        real_mask = mask[1]
        imag_mask = mask[2]

        # feature [B,C,F,T]
        mag = torch.abs(feature)
        pha = self.custom_atan2(feature.imag, feature.real)

        # stage 1
        # v83
        #mag = mag * F.softplus(mag_mask)
        # v84
        mag = mag * F.sigmoid(mag_mask)

        # stage 2
        mag_mask = torch.sqrt(torch.clamp(real_mask**2+imag_mask**2, eps))
        pha_mask = self.custom_atan2(imag_mask+eps, real_mask+eps)

        # default
        #real = mag * mag_mask.relu() * torch.cos(pha+pha_mask)
        #imag = mag * mag_mask.relu() * torch.sin(pha+pha_mask)

        # v85 sigmoid
        real = mag * mag_mask.sigmoid() * torch.cos(pha+pha_mask)
        imag = mag * mag_mask.sigmoid() * torch.sin(pha+pha_mask)
        return torch.stack([real, imag], dim=-1)
    
class fusion_cat(nn.Module):
    def __init__(self):
        super(fusion_cat,self).__init__()
    
    def forward(self,spec,face):
        return torch.cat([spec,face],dim=1)
    
class fusion_att(nn.Module):
    def __init__(self,embed_dim, num_heads=8):
        super(fusion_att,self).__init__()
        self.att = nn.MultiheadAttention(embed_dim,num_heads)
    
    def forward(self,spec,face):

        # B, C, F', T'
        B,C,F,T = spec.shape

        # B*T, C*F
        spec = spec.permute(0,3,1,2)
        face = face.permute(0,3,1,2)

        spec = spec.reshape(B*T,C*F)
        face = face.reshape(B*T,C*F)


        y,_ = self.att(spec,face,spec)

        y = y.reshape(B,T,C,F)
        y = y.permute(0,2,3,1)

        return y



class MMSUNet(nn.Module):
    def __init__(self, 
                 architecture,
                 input_channels=8,
                 n_fft=512,
                 use_face = True,
                 #model_complexity=45,
                 model_complexity=45,
                 bottleneck="None",
                 padding_mode="zeros",
                 type_masking = "CRM",
                 activation = "LeakyReLU",
                 dropout=0.0,
                 type_norm = "BatcNorm2d",
                 type_encoder = "Encoder",
                 type_residual = "None",
                 type_fusion = "cat"
                 ):
        super().__init__()

        self.bottleneck = bottleneck

        self.n_channel = input_channels
        n_angle = 2
        self.use_face = use_face

        print("MMSUNet::complexity {}".format(model_complexity))

        self.dropout = dropout

        ## Encoder
        self.encoders = []
        if type_encoder == "Encoder" : 
            module_cls = Encoder
        elif type_encoder == "EncoderDWN" : 
            module_cls = EncoderDWN
        elif type_encoder == "GConv" : 
            module_cls = GConv
        else :
            raise Exception("Not Implemented Encoder")
        
        self.n_layer= len(architecture["encoder"])

        for i in range(self.n_layer):
            module = module_cls(**architecture["encoder"]["enc{}".format(i+1)])
            self.add_module("encoder{}".format(i), module)
            self.encoders.append(module)

        if use_face :
            ## Facial feature encoder
            self.face_encoders = []
            for i in range(len(architecture["face_encoder"])) : 
                module = Encoder(**architecture["face_encoder"]["fenc{}".format(i+1)])
                self.add_module("face_encoder{}".format(i), module)
                self.face_encoders.append(module)
            self.face_encoders = nn.ModuleList(self.face_encoders)

        


        ## Decoder
        self.decoders = []
        if type_encoder == "Encoder" : 
            module_cls = Decoder
        elif type_encoder == "EncoderDWN" : 
            module_cls = DecoderDWN
        elif type_encoder == "GConv" : 
            module_cls = TrGConv
        else :
            raise Exception("Not Implemented Decoder")
        for i in range(self.n_layer):
            module = module_cls(**architecture["decoder"]["dec{}".format(self.n_layer - i)])
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)

        # Bottleneck
        if bottleneck == "TGRU" : 
            self.BTN = TGRUBlock(128,128,128)
        else :
            self.BTN = nn.Identity()

        if type_fusion == "cat" :
            self.fusion = fusion_cat()
        elif type_fusion == "att" :
            self.fusion = fusion_att(architecture["fusion"]["embed_dim"],architecture["fusion"]["num_heads"])
        else : 
            self.fusion = nn.Identity()

        # Reisuadl Connection
        self.res = []
        for i in range(self.n_layer-1) :
            if type_residual == "SE" :
                module = SEBlock(architecture["encoder"]["enc{}".format(i+1)]["out_channels"])
            else :
                module = nn.Identity()
            self.add_module("res{}".format(i), module)
            self.res.append(module)
            
        ## Attractor
        self.Attractor = Attractor(n_ch=4,n_fft=n_fft)

        self.attractEncoders = [] 
        for i in range(self.n_layer) : 
            module = AttractEncoder(
                n_ch= architecture["encoder"]["enc{}".format(i+1)]["out_channels"]
            )
            self.add_module("attractEncoder_{}".format(i),module)
            self.attractEncoders.append(module)


        conv = nn.Conv2d
        linear = conv(architecture["decoder"]["dec1"]["out_channels"], 2, 1)

        ## Mask Estimator
        self.add_module("linear", linear)
        self.padding_mode = padding_mode

        if type_masking == "CRM" : 
            self.mask = CRM()
        elif type_masking == "CRM_sig" : 
            self.mask = CRM_sig()
        elif type_masking == "CRM_soft" :
            self.mask = CRM_soft()
        elif type_masking == "CMEA" : 
            self.mask = CMEA()
        elif type_masking == "CM" : 
            self.mask = CM()
        else :
            raise Exception("Not Implemented")

        self.dr = nn.Dropout(self.dropout)

        self.decoders = nn.ModuleList(self.decoders)
        self.encoders = nn.ModuleList(self.encoders)
        self.attractors = nn.ModuleList(self.attractEncoders)
        self.res= nn.ModuleList(self.res)

        #self.lip = LipNet()

    def forward(self, sf,SV,theta,face):        
        # ipnut : [ Batch Channel Freq Time 2]

        attract = self.Attractor(SV,theta)
        #print("attract : {}".format(attract.shape))

        # Encoders
        sf_skip = []
        for i, encoder in enumerate(self.encoders):
            if i != self.n_layer - 1:
                sf_skip.append(self.res[i](sf))
            else : 
                sf_skip.append(sf)
            sf = encoder(sf)
            sf = self.dr(sf)
            #print("sf{}".format(i), sf.shape)

            a_s = self.attractEncoders[i](attract)
            a_s = torch.reshape(a_s,(*a_s.shape,1,1))
            #print("a_s{} : {}".format(i, a_s.shape))
            sf = a_s*sf
        # sf_skip : sf0=input sf1 ... sf9

        #face = self.lip(face)
        # Facial feature encoder
        if self.use_face : 
            for i, encoder in enumerate(self.face_encoders):
                face = encoder(face)
            p = self.fusion(sf,face)
        else :
            p = sf

        # Bottleneck
        if self.bottleneck == "RNN_AS" or self.bottleneck == "AATT": 
            p = self.BTN(p,attract)
        else : 
            p = self.BTN(p)

        # Decoders
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            if i == self.n_layer - 1:
                break
            #print(f"p{i}, {p.shape} + sf{self.n_layer - 1 - i}, {sf_skip[self.n_layer - 1 -i].shape}")
            
            p = torch.cat([p, sf_skip[self.n_layer - 1 - i]], dim=1)

        #print(p.shape)
        p = self.linear(p)
        # p : [B,1,F,T,2]
        M = self.mask(p)
        return M
    
    def masking(self,X,M):
        return self.mask.output(X,M)


class MMSUNet_helper(nn.Module):
    def __init__(self,
                 architecture,
                 n_fft = 512,
                 dropout = 0.0,
                 use_face = True,
                 bottleneck = "None",
                 model_complexity = 45,
                 use_SV = True,
                 mag_phase = False,
                 corr = False,
                 DSB = False,
                 activation= "LeakyReLU",
                 type_norm ="BatchNorm2d",
                 type_masking="CRM",
                 type_encoder = "Encoder",
                 type_residual = "None",
                 type_fusion = "cat"
                 ):
        super(MMSUNet_helper,self).__init__()

        self.n_fft = n_fft
        self.n_hfft = n_fft // 2 + 1

        self.n_channel = 4

        model_channel = self.n_channel*2

        if DSB :
            model_channel += 2

        self.mag_phase = mag_phase

        self.use_SV = use_SV
        self.corr = corr 
        self.DSB = DSB

        self.net = MMSUNet(
                        architecture=architecture,
                        input_channels = model_channel,
                        use_face = use_face,
                        dropout = dropout,
                        bottleneck=bottleneck,
                        model_complexity=model_complexity,
                        activation=activation,
                        type_norm = type_norm,
                        type_masking=type_masking,
                        type_encoder=type_encoder,
                        type_residual=type_residual,
                        type_fusion = type_fusion
                        )
        
        # const
        self.sr = 16000
        self.ss = 340.4
        self.dist = 100

        self.window = torch.hann_window(self.n_fft)

    def forward(self,x,angle,mic_pos,face):
        B,C,L = x.shape

        x = torch.reshape(x,(B*C,L))
        X = torch.stft(x,n_fft = self.n_fft,window=self.window.to(x.device),return_complex=True)
        _, F,T = X.shape
        short = 16-T%16
        X = torch.nn.functional.pad(X,(0,short))
        _, F,T = X.shape
        X = X.reshape(B,C,F,T)

        if self.corr : 
            for i in range(B) : 
                for j in range(C-1) : 
                    X[i,j+1] = X[i,0] * X[i,j+1]

        SV = self.steering_vector(angle,mic_pos)
        SV = torch.cat((SV.real,SV.imag),-1)

        theta = self.anlge_pre(angle)
        #print("angle feaute : {}".format(angle_feature.shape))

        # [B,C,F,T,2]
        spectral_feature = torch.cat((X.real,X.imag),dim=1)

        if self.DSB : 
            # B,F,T
            DSB = self.delay_n_sum(X,angle,mic_pos)
            # B,2,F,T
            DSB = torch.stack((DSB.real,DSB.imag),dim=1)

            # B,C+1,F,T,2
            spectral_feature = torch.cat((spectral_feature,DSB),dim=1)

        mask = self.net(spectral_feature,SV,theta,face)

        #print("mask {} | X {}".format(mask.shape,X.shape))

        Y = self.net.masking(X[:,0],mask)
        y = torch.istft(Y,n_fft = self.n_fft,window=self.window.to(Y.device),length=L)
        return y

    def anlge_pre(self,angle):
        sin_theta = torch.sin((-angle)/180*torch.pi)
        cos_theta = torch.cos((-angle)/180*torch.pi)
        return torch.stack((sin_theta,cos_theta),1)
    
    def steering_vector(self,angle,mic_pos) :
        """
        angle : [B]
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
    
    # Delay-and-Sum Beamformer
    def delay_n_sum(self,X,angle,mic_pos):
        """
        angle : [B]
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

        const = 1j*2*torch.pi*self.sr/(self.n_fft*self.ss)
        h  = torch.zeros(B,self.n_channel,self.n_hfft,dtype=torch.cfloat).to(mic_pos.device)
        for i in torch.arange(B) : 
            for j in range(self.n_hfft) : 
                h[i,:,j] = torch.exp(j*TDOA[i]*const)
                h[i,:,j] /= torch.norm(h[i,:,j])

        h = torch.unsqueeze(h,-1)

        Y = torch.einsum('bcft,bcfl->bft', [X, h])
        return Y

def test() : 
    B = 2
    C = 4
    F = 257
    L = 32000
    T = 256

    x = torch.rand(B,C,L)
    angle = torch.rand(B)
    mic_pos = torch.rand(B,4,3)

    m = MMSUNet_helper(type_masking="CRM")

    y = m(x,angle,mic_pos)

    print("output : {}".format(y.shape))