import torch



# Log-Power Spectral
def LogPowerSpectral(stft):
    # dB scale
    power_spectral = torch.abs(stft)
    log_power_spectral = 10*torch.log10(1+power_spectral)
    return log_power_spectral

def Mag(stft):
    mag = torch.abs(stft)
    return mag

""" Inter-phase difference
Difference with first channel
(2018)Recognizing Overlapped Speech in Meetings: A Multichannel Separation Approach Using Neural Networks
https://arxiv.org/pdf/1810.03655.pdf

+ input
stft : [C, F, T, X]
+ return
IPD : [C-1, F, T]

"""
def InterPhaseDifference(stft,full=False):
    d = stft[1:,:,:]/(stft[0,:,:]+1e-13)
    
    if full :
        for i in range(1,stft.shape[0]) :
            t = stft[i+1:,:,:]/(stft[i,:,:]+1e-13)

            d = torch.cat((d,t),dim=0)

    IPD =  torch.angle(d)
    return IPD


def NormalizedIPD(stft):
    raise Exception("NormalizedIPD::Not Implemented")

def cosIPD(stft,full=False):
    IPD =  InterPhaseDifference(stft,full)
    return torch.cos(IPD)

def sinIPD(stft,full=False):
    IPD =  InterPhaseDifference(stft,full)
    return torch.sin(IPD)

def cossinIPD(stft,full=False):
    cos = cosIPD(stft,full)
    sin = sinIPD(stft,full)

    return torch.concat((cos,sin))








"""Angle Feauture
To get the angle feature, we first form the steering vector
for the direction-of-arrival (DOA) of each speaker. Then, the cosine
distance between the steering vector and the complex spectrum of
each channel that is normalized with respect to the first microphone 
is calculated as follows:

\begin{equation*}A_{n,tf} = \sum\limits_{i = 1}^M \frac{e_n^{i,f} \frac{y_{i,tf}}{y_{1,tf}}}{\left| e_n^{i,f} \frac{y_{i,tf}}{y_{1,tf}} \right|}\tag{2}\end{equation*}

where n is the speaker index, e^{i,f}_n is the steering vector coefficient
for speaker n’s DOA at microphone i and frequency bin f.

Intuitively, the angle feature lets the network to attend the sound coming
from the direction of a certain speaker. The idea of the angle feature
is similar to beamforming while it is different from the traditional
beamforming in that non-linear processing is performed with a neural network.

+ input
stft  : [C, F, T, 2(complex) ]
angle : [N(target), T, 2(azimuth,elevation)] 
mic_pos : [C,3]

+ return
AF : [N,C,F,T]
"""
def AngleFeature(stft,angle,mic_pos,fs=16000,complex=False,dist=1.0):
    # F : n_hfft
    C,F,T = stft.shape 
    N,_,_ = angle.shape

    ss = 340.3
    pi = 3.141592653589793
    n_fft = 2*F-2
    #n_fft = 2*F+1


    ## location of sources
    loc_src = torch.zeros(N,T,3)
    loc_src[:,:,0] = dist*torch.cos((90-angle[:,:,0])/180*pi)*torch.sin((90-angle[:,:,1])/180*pi)
    loc_src[:,:,1] = dist*torch.sin((90-angle[:,:,0])/180*pi)*torch.sin((90-angle[:,:,1])/180*pi)
    loc_src[:,:,2] = dist*torch.cos((90-angle[:,:,1])/180*pi)

    # TDOA
    TDOA = torch.zeros(N,C,T)

    for i in range(C) : 
        TDOA[:,i,:] = torch.norm(mic_pos[i,:] - loc_src[:,:,:])

    ## Steering vector
    SV = torch.zeros(N,C,F,T, dtype=torch.cfloat)
    for i in range(F):
        SV[:,:,i,:] = torch.exp(-1j*2*pi*i/n_fft*TDOA*fs/ss)
        #[M] st(:,freq) = st(:,freq).*sqrt(nSensor/norm(st(:,freq)))

        # normalization per channel
        for i_N in range(N) : 
            for i_T in range(T) : 
                SV[i_N,:,i,i_T] = SV[i_N,:,i,i_T]/torch.sqrt((C)/torch.norm(SV[i_N,:,i,i_T]))
        #SV[:,:,i,:] = torch.exp(1j*2*pi*i/n_fft*TDOA*fs/ss)
    # norm

    """
    K. Tan, Y. Xu, S. Zhang, M. Yu and D. Yu, "Audio-Visual Speech Separation and Dereverberation With a Two-Stage Multimodal Network," in IEEE Journal of Selected Topics in Signal Processing, vol. 14, no. 3, pp. 542-553, March 2020, doi: 10.1109/JSTSP.2020.2987209.

    ... complex-valued, and they are treated as 2-D vectors in the operations < ·, · > and ||·||, where their real and imaginary parts are regarded as two vector  components.
    """

    ## Angle Feature
    if complex : 
        AF = torch.zeros(N,C,F,T, dtype=torch.cfloat)
        for i in range(N) : 
            tmp_term =  SV[i,:,:,:]*(stft[:,:,:]/stft[0:1,:,:])
            AF[i,:,:,:] = tmp_term/torch.abs(tmp_term)
        AF = torch.sum(AF,axis=1)
    else :
        ## real, imag as independent channel
        #AF = torch.zeros(2*N,C,F,T, dtype=torch.cfloat)
        #SV = torch.view_as_real(SV)
        #for i in range(N) : 
        #    tmp_stft = stft[:,:,:]/stft[0:1,:,:]
        #    tmp_stft = torch.view_as_real(tmp_stft)
        #    tmp_term = SV[i,:,:,:,:]*tmp_stft[:,:,:,:] 
        #    AF[2*i,:,:,:] = tmp_term[:,:,:,0]/torch.ans(tmp_term[:,:,:,0])
        #    AF[2*i+1,:,:,:] = tmp_term[:,:,:,1]/torch.ans(tmp_term[:,:,:,1])
        #AF = torch.sum(AF,axis=1)

        ## split real,imag in AF only
        AF = torch.zeros(N,C,F,T, dtype=torch.cfloat)
        for i in range(N) : 
            tmp_term =  SV[i,:,:,:]*(stft[:,:,:]/(stft[0:1,:,:]+1e-13))
            AF[i,:,:,:] = tmp_term/(torch.abs(tmp_term)+1e-13)
        AF = torch.sum(AF,axis=1)
        AF = torch.view_as_real(AF)                
        AF = torch.reshape(AF,(2*N,F,T))

    return AF

"""
(2021,arXiv)SALSA : Spatial Cue-Augmented Log-Spectrogram Features for Polyphonic Sound Event Localization and Detection
https://arxiv.org/pdf/2110.00275.pdf
"""
def SALSA():
    pass


"""
(2022,ICASSP)SALSA-LITE: A FAST AND EFFECTIVE FEATURE FOR POLYPHONIC SOUND EVENT LOCALIZATION AND DETECTION WITH MICROPHONE ARRAYS
https://arxiv.org/pdf/2111.08192.pdf
"""
def SALSA_lite(stft,mic_pos,speed_of_sound=340.3):
    pass