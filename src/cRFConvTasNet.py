"""
 2022-03 ~ 
 kooBH
 https://github.com/kooBH

 Conv-TasNet based cRF estimator for implementation for
  (2021,INTERSPEECH)MIMO Self-Attentive RNN Beamformer for Multi-Speaker Speech Separation
 https://www.isca-speech.org/archive/interspeech_2021/li21c_interspeech.html
 
 Using modules from  
   https://github.com/kaituoxu/Conv-TasNet
   Created on 2018/12
   Author: Kaituo XU
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-13

"""
    following descriptions of 
    Z. Zhang et al., "Multi-Channel Multi-Frame ADL-MVDR for Target Speech Separation," in IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 3526-3540, 2021, doi: 10.1109/TASLP.2021.3129335.
    https://ieeexplore.ieee.org/document/9623492
"""
class cRFConvTasNet(nn.Module):
    def __init__(self,
    L_t=1,
    L_f=1,
    c_in=4,
    n_target=4,
    f_ch=256,
    n_fft=512,
    mask="Sigmoid",
    TCN_activation="None"
    ):
        """
            L : length of cRF
            f_ch : feature channel, default 256
        """
        super(cRFConvTasNet,self).__init__()


        ## Audio Encoding Network
        # to reduce feature dimmension
        # 256 1x1 kernel conv
        n_hfft = int(n_fft/2 + 1)
        
        dim_input = (1 + c_in-1 + 2*n_target) * n_hfft
        input_layer = nn.Conv1d(dim_input,f_ch,1)
        self.N = n_target

        print("{}::dim_input : {}".format(self.__class__.__name__,dim_input))

        self.C = c_in
        self.L_t = L_t
        self.L_f = L_f
        self.F = n_hfft

        # stack of two successive TCN bolck 2^0 to 2^7 dialation
        encoder = TCN(
            c_in = f_ch,
            c_out= f_ch * 2,
            kernel = 3,
            n_successive =2,
            n_block = 8,
            TCN_activation=TCN_activation
        )

        ## separate filter estimatior network
        # two successive TCN blcoks 2^0 to 2^7 dialation
        filter_estimator = TCN(
            c_in = f_ch,
            c_out= f_ch * 2,
            kernel = 3,
            n_successive =2,
            n_block = 8,
            TCN_activation=TCN_activation
        )
        """
        + Complex Ratio Filter
        W. Mack and E. A. P. Habets, "Deep Filtering: Signal Extraction and Reconstruction Using Complex Time-Frequency Filters," in IEEE Signal Processing Letters, vol. 27, pp. 61-65, 2020, doi: 10.1109/LSP.2019.2955818.
        """ 
        ## output
        # L = 0 : cBM
        # L > 0 : cRF
        # n_target * n_channel * freq * filter * complex 
        dim_output = n_target * c_in * n_hfft * ((2*L_t+1)*(2*L_f+1)) * 2
        conv_output = nn.Conv1d(
            f_ch,
            dim_output,
            1
        )

        if mask == "Sigmoid" : 
            activation_output= nn.Sigmoid()
        elif mask == "Softplus":
            activation_output= nn.Softplus()
        elif mask == "Tanh" : 
            activation_output = nn.Tanh()
        else :
            raise Exception("ERROR::{}:Unknown type of mask : {}".format(__name__,mask))

        self.net = nn.Sequential(
            input_layer,
            encoder,
            filter_estimator,
            conv_output,
            activation_output
            )        
        
        self.net.apply(init_weights)

    # input : (B, Flatten, T)
    def forward(self,x):
        B = x.shape[0]
        T = x.shape[-1]
        filter = self.net(x)

       # [B,N, C, filter, n_hfft, Time, 2(complex)]
        filter = torch.reshape(filter,(B,self.N,self.C, (2*self.L_f+1),(2*self.L_t+1),self.F,T,2))

        # Return in Complex type
        return torch.view_as_complex(filter)

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Temporal Convolution Network Block
class TCN(nn.Module):
    def __init__(self, 
    c_in=256, 
    c_out=512, 
    kernel=3, 
    n_successive=2, 
    n_block=8, 
    norm_type="gLN", 
    causal=True,
    mask='relu',
    TCN_activation="None"
    ):
        """
        Args:
            c_in : 
            c_out : 
            kernel : Kernel size in convolutional separate blocks
            n_succesive : Number of convolutional blocks in each repeat
            n_block : Number of repeats

            norm_type: BN, gLN, cLN
            causal: causal or non-causal
        """
        super(TCN, self).__init__()

        repeats = []
        for r in range(n_successive):
            blocks = []
            for x in range(n_block):
                dilation = 2**x
                padding = (kernel - 1) * dilation if causal else (kernel - 1) * dilation // 2
                blocks += [TemporalBlock(
                    in_channels=c_in,
                    out_channels= c_out,
                    kernel_size= kernel, 
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    norm_type=norm_type,
                    causal=causal)]
            repeats += [nn.Sequential(*blocks)]

        self.net = nn.Sequential(*repeats)
        if TCN_activation == "None" : 
            self.activation = nn.Identity()
        elif TCN_activation == "Tanh" : 
            self.activation == nn.Tanh()
        elif TCN_activation == "Sigmoid" : 
            self.activation = nn.Sigmoid()
        elif TCN_activation == "PReLU" : 
            self.activation = nn.PReLU()
        else : 
            raise Exception("ERROR::TCN:: {} is invalid activation".format(TCN_activation))

    def forward(self, x):
        return self.activation(self.net(x)) 
 


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, norm_type="gLN", causal=False):
        super(TemporalBlock, self).__init__()
        # [M, B, K] -> [M, H, K]
        conv1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, out_channels)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(out_channels, in_channels, kernel_size,
                                        stride, padding, dilation, norm_type,
                                        causal)
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):
        """
        Args:
            x: [M, B, K]
        Returns:
            [M, B, K]
        """
        residual = x
        out = self.net(x)
        # TODO: when P = 3 here works fine, but when P = 2 maybe need to pad?
        return out + residual  # look like w/o F.relu is better than w/ F.relu
        # return F.relu(out + residual)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, norm_type="gLN", causal=False):
        super(DepthwiseSeparableConv, self).__init__()
        # Use `groups` option to implement depthwise convolution
        # [M, H, K] -> [M, H, K]
        depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   dilation=dilation, groups=in_channels,
                                   bias=False)
        if causal:
            chomp = Chomp1d(padding)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, in_channels)
        # [M, H, K] -> [M, B, K]
        pointwise_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        # Put together
        if causal:
            self.net = nn.Sequential(depthwise_conv, chomp, prelu, norm,
                                     pointwise_conv)
        else:
            self.net = nn.Sequential(depthwise_conv, prelu, norm,
                                     pointwise_conv)

    def forward(self, x):
        """
        Args:
            x: [M, H, K]
        Returns:
            result: [M, B, K]
        """
        return self.net(x)


class Chomp1d(nn.Module):
    """To ensure the output length is the same as the input.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Args:
            x: [M, H, Kpad]
        Returns:
            [M, H, K]
        """
        return x[:, :, :-self.chomp_size].contiguous()


def chose_norm(norm_type, channel_size):
    """The input of normlization will be (M, C, K), where M is batch size,
       C is channel size and K is sequence length.
    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size)
    else: # norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)


# TODO: Use nn.LayerNorm to impl cLN to speed up
class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)"""
    def __init__(self, channel_size):
        super(ChannelwiseLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, N, K]
        """
        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size,1 ))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True) #[M, 1, 1]
        var = (torch.pow(y-mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


if __name__ == "__main__":
    B = 2
    C = 4
    N = 2
    L = 1
    T = 250
    F = 257

    net = cRFConvTasNet(
        L_t=L,
        L_f=L,
        n_target = N
    )
    x = torch.rand(B,(C + N*2)*F,T)

    y = net(x)
    print(y.shape)
