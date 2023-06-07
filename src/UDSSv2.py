"""
UNet based Directional Source Separation
"""

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,ch_in,ch_out,kernel,n_dim, stride,padding=None,activation="PReLU"):
        super(Encoder,self).__init__()

        if padding is None:
            padding = [(i - 1) // 2 for i in kernel]  # 'SAME' padding

        ## SF
        self.conv = nn.Conv2d(ch_in,ch_out,kernel,stride,padding)
        self.norm = nn.LayerNorm(ch_out)
        self.act = activation

        ## AF
        self.angle = nn.Linear(2, n_dim)
        self.act_AF = nn.Sigmoid()

    def forward(self, SF, AF):
        
        s = self.conv(SF)
        s = self.norm(s)
        s = self.act(s)

        a = self.angle(AF)
        a = self.act_AF(a)

        # apply angle mask

        s = s * a

class Decoder(nn.Module) : 
    def __init__(self, ch_in,ch_out,kernel, stride,padding=None,activation="PReLU"):
        super(Decoder,self).__init__()

        self.tconv = nn.ConvTranspose2d(ch_in,ch_out,kernel,stride,padding)
        self.norm = nn.LayerNorm(ch_out)
        self.act = activation

    def forward(self, x):
        x = self.tconv(x)
        x = self.norm(x)
        x = self.act(x)
            
        return x
    
default_UDSSv2_architecutere = {
    "Encoder":[],
    "Decoder":[]
}
    
class UDSSv2(nn.Module  ) : 
    def __init__(self): 
        super(UDSSv2, self).__init__()



    def forward(self,SF,AF) : 
        pass


class UDSSv2_helper(nn.Module):
    def __init__(self):
        super(UDSSv2_helper, self).__init__()

    def forward(self,x,theta):
        pass



if __name__ == "__main__":
    m = UDSSv2()

    B = 2
    C = 4
    F = 257
    T = 40

    SF = torch.randn(B,C,F,T)
    AF = torch.randn(B,2,T)

    y = m(SF,AF)
    print(y.shape)