import torch



# Log-Power Spectral
def LogPowerSpectral(stft):
    # dB scale
    power_spectral = torch.abs(stft)
    log_power_spectral = 10*torch.log10(power_spectral)
    return log_power_spectral

""" Inter-phase difference
Difference with first channel
(2018)Recognizing Overlapped Speech in Meetings: A Multichannel Separation Approach Using Neural Networks
https://arxiv.org/pdf/1810.03655.pdf

+ input
stft : [C, F, T, X]
+ return
IPD : [C-1, F, T]

"""
def InterPhaseDifference(stft):
    d = stft[1:,:,:]/stft[0,:,:]
    IPD =  torch.angle(d)
    return IPD


def NormalizedIPD(stft):
    raise Exception("NormalizedIPD::Not Implemented")

def cosIPD(stft):
    d = stft[1:,:,:]/stft[0,:,:]
    IPD =  torch.angle(d)
    return torch.cos(IPD)

def sinIPD(stft):
    d = stft[1:,:,:]/stft[0,:,:]
    IPD =  torch.angle(d)
    return torch.sin(IPD)


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
def AngleFeature(stft,angle,mic_pos,fs=16000):
    # F : n_hfft
    C,F,T,_ = stft.shape 
    N,_,_ = angle.shape

    ss = 340.3
    pi = 3.141592653589793
    n_fft = 2*F+1

    ## relative_distance_of_arrival
    d_angle = torch.zeros(N,T,3)
    d_angle[:,:,0] = torch.cos(angle[:,:,0]/180*pi)*torch.sin(angle[:,:,1]/180*pi)
    d_angle[:,:,1] = torch.sin(angle[:,:,0]/180*pi)*torch.sin(angle[:,:,1]/180*pi)
    d_angle[:,:,2] = torch.cos(angle[:,:,1]/180*pi)

    d_dist = mic_pos[0:1,:] - mic_pos[:,:]

    RDOA = torch.zeros(N,C,T)
    for i in range(N) : 
        RDOA[i,:,:] = torch.matmul(d_dist,d_angle[i,:,:].T) 

    ## steering vector
    SV = torch.zeros(N,C,F,T)
    for i in range(F):
        SV[:,:,i,:] = torch.exp(1j*2*pi*i/n_fft*RDOA*fs/ss)

    ## Angle Feature
    AF = torch.zeros(N,C,F,T)
    for i in range(N) : 
        tmp_term =  steering_vector[i,:,:,:]*(stft[:,:,:]/stft[0:1,:,:])
        AF [i,:,:,:] = tmp_term/torch.abs(tmp_term)
    AF = torch.sum(AnlgeFeature,axis=2)

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