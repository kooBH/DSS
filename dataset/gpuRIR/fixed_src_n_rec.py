#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Example for computing the RIR between several sources and receivers in GPU.
"""

import numpy as np
from math import ceil
from scipy.io import wavfile

import gpuRIR
gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(True)

fs, source_signal = wavfile.read('source_signal.wav')
print("source signal : " + str(source_signal.shape))

room_sz = [5,5,2.5]  # Size of the room [m]
nb_src = 1  # Number of sources
pos_src = np.array([[1,2.9,0.5]]) # Positions of the sources ([m]
nb_rcv = 4 # Number of receivers
pos_rcv = np.array([
    [2.5 + 0.00, 2.5 + 0.00, 0.5],
    [2.5 + 0.08, 2.5 + 0.00, 0.5],
    [2.5 + 0.00, 2.5 + 0.08, 0.5],
    [2.5 + 0.08, 2.5 + 0.08, 0.5],
    ])	 # Position of the receivers [m]


mic_pattern = "omni" # Receiver polar pattern
orV_rcv=None # None for omni

abs_weights = [0.9]*5+[0.5] # Absortion coefficient ratios of the walls

T60 = 0.7	 # Time for the RIR to reach 60dB of attenuation [s]
att_diff = 15.0	# Attenuation when start using the diffuse reverberation model [dB]
att_max = 60.0 # Attenuation at the end of the simulation [dB]

beta = gpuRIR.beta_SabineEstimation(room_sz, T60, abs_weights=abs_weights) # Reflection coefficients
Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, T60) # Time to start the diffuse reverberation model [s]
Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)	 # Time to stop the simulation [s]
nb_img = gpuRIR.t2n( Tdiff, room_sz )	# Number of image sources in each dimension
RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern)

t = np.arange(int(ceil(Tmax * fs))) / fs

print("RIRs : " + str(RIRs.shape))

output = []
for i in range(nb_rcv) : 
    output.append(np.convolve(source_signal,RIRs[0,i,:]))
    print(np.shape(output)) 
output = np.stack(output,axis=1)
print(output.shape)
output= output/np.max(np.abs(output))
wavfile.write("filtered_signal.wav",fs,output)