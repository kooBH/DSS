import torch
import torch.nn as nn

from Conformer import Conformer

class DSS(nn.Module):
    def __init__(self, type_separtor, n_channel=5):
        super(DSS, self).__init__()
        
        self.n_fft = 512
        self.n_hfft = 257

        if type_separtor == "Conformer" : 
            self.separator = Conformer(
              257,
              num_spk=1,
              input_ch = n_channel,
              input_layer = "conv2d"
            )
            self.do_masking = True
        else :
            raise Exception("ERROR::Unknown separator : {}".format(type_separtor))

        ## For inference
        self.window = torch.hann_window(self.n_fft)

    # x : [B, C, F, T]
    def forward(self,x):
       out = self.separator(x)
       return out

    def output(self, noisy,estim,hp=None) : 
        self.window = self.window.to(estim.get_device())
        B,C,L = noisy.shape

        # masking
        if self.do_masking : 
            B,_,F,T = estim.shape
            spec = torch.stft(noisy[:,0,:],n_fft=self.n_fft,window=self.window,return_complex=True)[:,:,:T]

            estim = estim[:,0,:,:] + 1j*estim[:,1,:,:]

            output = spec * estim
        # linear filter
        else :
            noisy = torch.reshape(noisy,(B*C,L))

            B,C,F,T = filter.shape

            spec = torch.stft(noisy,n_fft=self.n_fft,window=self.window,return_complex=True)

            spec = torch.reshape(spec,(B,C,F,T))

            spec = torch.permute(spec,(0,2,3,1))
            estim = torch.permute(estim,(0,2,3,1))

            spec = torch.reshape(spec,(B*F*T,1,C))
            estim = torch.reshape(estim,(B*F*T,C,1))

            # [B*F*T,1,1]
            output = torch.bmm(spec,estim)
            output = torch.reshape(output,(B,F,T))

        output = torch.istft(output,n_fft=512,window=self.window,length=L)

        return output