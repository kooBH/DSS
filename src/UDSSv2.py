"""
UNet based Directional Source Separation
"""
import torch
import torch.nn as nn

class Conv2dBlock(nn.Module):
    def __init__(self,ch_in,ch_out,kernel,stride,padding=None,activation="PReLU"):
        super(Conv2dBlock,self).__init__()

        if padding is None:
            padding = [(i - 1) // 2 for i in kernel]  # 'SAME' padding

        self.conv = nn.Conv2d(ch_in,ch_out,kernel,stride,padding)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.PReLU() 

    def forward(self, s):
        s = self.conv(s)
        s = self.norm(s)
        s = self.act(s)
        return s

class TrConv2dBlock(nn.Module) : 
    def __init__(self, ch_in,ch_out,kernel, stride,padding=None,activation="PReLU"):
        super(TrConv2dBlock,self).__init__()

        self.tconv = nn.ConvTranspose2d(ch_in,ch_out,kernel,stride,padding)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.tconv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class DepthSepNextBlock(nn.Module) : 
    def __init__(self,ch_in,ch_out,kernel,stride,padding=None,activation="PReLU"):
        super(DepthSepNextBlock,self).__init__()

        if padding is None:
            padding = [(i - 1) // 2 for i in kernel]  # 'SAME' padding

        self.conv_depth = nn.Conv2d(ch_in,ch_in,kernel,stride,padding,groups=ch_in)
        self.norm_depth = nn.BatchNorm2d(ch_in)

        self.conv_point = nn.Conv2d(ch_in,ch_out,kernel_size=1)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.conv_depth(x)
        x = self.norm_depth(x)

        x = self.conv_point(x)
        x = self.act(x)
        return x
    
class Attractor(nn.Module) :
    def __init__(self,n_ch=4,n_fft=512,n_dim=None):
        super(Attractor,self).__init__()

        if n_dim is None : 
            n_dim = n_fft//2 +1

        self.enc_SV = nn.Linear(n_ch *(n_fft+2), n_dim)
        self.enc_theta = nn.Linear(2,n_dim)

        self.act_SV = nn.PReLU()
        self.act_theta = nn.Sigmoid()

        self.bn = nn.BatchNorm1d(n_dim)

        self.enc_2 = nn.Linear(n_dim, n_dim*2)
        self.act_2 = nn.PReLU()
        self.enc_3 = nn.Linear(n_dim*2, n_fft//2 +1)
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
    def __init__(self,n_ch, n_dim,n_fft = 512):
        super(AttractEncoder,self).__init__()

        self.linear = nn.Linear(n_fft//2+1, n_dim)
        self.conv = nn.Conv1d(1,n_ch,kernel_size=1)
        self.act = nn.Sigmoid()

        self.dr = nn.Dropout(0.2)

    def forward(self,attract):
        attract = self.linear(attract)
        attract = self.dr(attract)
        attract = attract.unsqueeze(1)
        attract = self.conv(attract)
        attract = self.act(attract)

        return attract
    
default_UDSSv2_architecutere = {
    # 
    "Encoder":[
        {"ch_in" : 4,"ch_out" : 8,"kernel":(7,1),"stride":(1,1),"padding":(3,0),"activation":"PReLU"},
        {"ch_in" : 8,"ch_out": 16,"kernel":(1,7),"stride":(1,1),"padding":(0,3),"activation":"PReLU"},
        {"ch_in" : 16,"ch_out" : 32,"kernel":(5,5),"stride":(2,2),"padding":(0,0),"activation":"PReLU"},
        {"ch_in" : 32,"ch_out" : 64,"kernel":(5,5),"stride":(1,2),"padding":(0,0),"activation":"PReLU"},
        {"ch_in" : 64,"ch_out" : 128,"kernel":(3,3),"stride":(2,2),"padding":(0,0),"activation":"PReLU"},
        {"ch_in" : 128,"ch_out" : 256,"kernel":(3,3),"stride":(2,2),"padding":(0,0),"activation":"PReLU"},
    ],

    # 
    "Decoder":[
        {"ch_in" : 256,"ch_out" : 128,"kernel":(5,4),"stride":(2,2),"padding":(1,0),"activation":"PReLU"},
        {"ch_in" : 128,"ch_out" : 64,"kernel":(5,3),"stride":(2,2),"padding":(1,0),"activation":"PReLU"},
        {"ch_in" : 64,"ch_out" : 32,"kernel":(5,6),"stride":(1,2),"padding":(0,0),"activation":"PReLU"},
        {"ch_in" : 32,"ch_out" : 16,"kernel":(5,6),"stride":(2,2),"padding":(0,0),"activation":"PReLU"},
        {"ch_in" : 16,"ch_out" : 8,"kernel":(3,3),"stride":(1,1),"padding":(1,1),"activation":"PReLU"},
        {"ch_in" : 8,"ch_out" : 1,"kernel":(3,3),"stride":(1,1),"padding":(1,1),"activation":"PReLU"},
               ],
    "AttractorEncoder": [
        {"n_ch": 8,"n_dim": 257},
        {"n_ch": 16,"n_dim": 257},
        {"n_ch": 32,"n_dim": 127},
        {"n_ch": 64,"n_dim": 123},
        {"n_ch": 128,"n_dim": 61},
        {"n_ch": 256,"n_dim": 30}
    ]
}
    
class UDSSv2(nn.Module) : 
    def __init__(self,architecture = default_UDSSv2_architecutere,
                 encoder = "Conv2d"
                 
                 ): 
        super(UDSSv2, self).__init__()

        if encoder == "Conv2d" : 
            encoder = Conv2dBlock
        elif encoder == "DepthSepNextBlock" :
            encoder = DepthSepNextBlock
        else :
            raise ValueError("Not supported encoder type : {}".format(encoder))

        self.Encoder = nn.ModuleList()
        for i in range(len(architecture["Encoder"])) :
            self.Encoder.append(encoder(**architecture["Encoder"][i]))

        self.Decoder = nn.ModuleList()
        for i in range(len(architecture["Decoder"])) :
            self.Decoder.append(TrConv2dBlock(**architecture["Decoder"][i]))

        self.Attractor = Attractor(n_dim=1024)
        self.AttractorEncoder = nn.ModuleList()
        for i in range(len(architecture["AttractorEncoder"])) :
            self.AttractorEncoder.append(AttractEncoder(**architecture["AttractorEncoder"][i]))

        self.bottleneck = nn.Identity()
        self.masking = nn.Sigmoid()
        self.dr = nn.Dropout(0.2)

    def forward(self,SF,AF,theta) : 
        # SF : [B,C,F,T,2]
        x = SF
        attractor = self.Attractor(AF,theta)

        # Encoder
        skip = []
        skip.append(x)
        for i in range(len(self.Encoder)) :
            x = self.Encoder[i](x)
            m = self.AttractorEncoder[i](attractor)
            m = torch.unsqueeze(m,-1)
            #print("Encoder[{}] : {} | {}".format(i,x.shape,m.shape))
            x = x * m
            skip.append(x)

        # Bottleneck
        x = self.bottleneck(x)
        x = self.dr(x)
        #print("Bottleneck : {}".format(x.shape))

        x = self.Decoder[0](x)
        #print("Decoder[{}] : {}".format(0,x.shape))
        # Encoder
        for i in range(1,len(self.Decoder)) :
            #print("x + skip[{}] : {} + {} ".format(len(skip)-(i+1),x.shape,skip[-(i+1)].shape))
            x = self.Decoder[i](x + skip[-(i+1)])

           # print("Decoder[{}] : {}".format(i,x.shape))

        mask = self.masking(x)

        return mask

class UDSSv2_helper(nn.Module):
    def __init__(self,
                 encoder = "Conv2d",
                 ):
        super(UDSSv2_helper, self).__init__()

        # const
        self.sr = 16000
        self.ss = 340.4
        self.dist = 100
        self.n_channel = 4

        self.n_fft = 512
        self.n_hfft = 512//2 + 1

        self.window = torch.hann_window(self.n_fft)

        self.net = UDSSv2(encoder=encoder)

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

    def forward(self,x,angle,mic_pos):
        B,C,L = x.shape

        x = torch.reshape(x,(B*C,L))
        X = torch.stft(x,n_fft = self.n_fft,window=self.window.to(x.device),return_complex=True)
        _, F,T = X.shape
        short = 16-T%16
        X = torch.nn.functional.pad(X,(0,short))
        _, F,T = X.shape
        X = X.reshape(B,C,F,T)

        AF = self.steering_vector(angle,mic_pos)
        AF = torch.cat((AF.real,AF.imag),-1)
        #AF = AF.unsqueeze(-2).expand(-1, -1, T, -1)

        theta = self.anlge_pre(angle)
        #print("angle feaute : {}".format(angle_feature.shape))

        #SF= torch.stack((X.real,X.imag),dim=-1)
        mag = torch.abs(X[:,0:1])
        phs = torch.angle(X)
        IPD = torch.stack((phs[:,0]-phs[:,1],phs[:,0]-phs[:,2],phs[:,0]-phs[:,3]),dim=1)
        SF = torch.cat((mag,IPD),dim=1)

        mask = self.net(SF,AF, theta)

        Y = X[:,0]*mask[:,0]
        y = torch.istft(Y,n_fft = self.n_fft,window=self.window.to(Y.device),length=L)
        return y

if __name__ == "__main__":
    B = 2
    C = 4

    """
    m = UDSSv2()
    F = 257
    T = 40
    SF = torch.randn(B,C,T,F)
    AF = torch.randn(B,F*2)
    theta = torch.randn(B,2)
    y = m(SF,AF,theta)
    """

    L = 16000

    x = torch.rand(B,C,L)
    theta = torch.rand(B)
    mic = torch.rand(B,C,3)

    m = UDSSv2_helper()
    y = m(x,theta,mic)


    print(y.shape)