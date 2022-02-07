#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Example for simulating the recording of a moving source with a microphone array.
You need to have a 'source_signal.wav' audio file to use it as source signal and it will generate
the file 'filtered_signal.wav' with the stereo recording simulation.
"""
import numpy as np


import gpuRIR
gpuRIR.activateMixedPrecision(False)

# TODO
'''
multiple sources
overlap
diffuse noise
'''

class RIR:
    def __init__(self
        ,room_size = [3, 3, 2.5] # (m)
        ,RT60=0.7 # (sec)
        ,pos_render = [1.5, 2.5, 1.0]  # (m)
        ,pos_capture = [1.5, 1.5, 1.0] # (m)
        ,vec_render = None
        ,vec_capure = None
        ,pts_traj= 100 # Number of trajectory points
        ,att_diff = 15.0  # Attenuation when start using the diffuse reverberation model (dB)
        ,att_max = 60.0 # Attenuation at the end of the simulation (dB)
    ):

    def run(self):


room_sz = [8,4,2.5]  # Size of the room [m]

## NOTE : discrete trajectory points 
traj_pts = 300  # Number of trajectory points

pos_traj = np.tile(np.array([0.0,3.0,1.0]), (traj_pts,1))

pos_traj[:,0] = np.linspace(0.1, 6, traj_pts) # Positions of the trajectory points [m]
nb_rcv = 2 # Number of receivers
pos_rcv = np.array([[4.00,1,1.5],[4.04,1,1.5]])	 # Position of the receivers [m]
orV_rcv = np.array([[-1,0,0],[1,0,0]])
mic_pattern = "card" # Receiver polar pattern
#T60 = 0.6 # Time for the RIR to reach 60dB of attenuation [s]
T60 = 0.7 # Time for the RIR to reach 60dB of attenuation [s]
att_diff = 15.0	# Attenuation when start using the diffuse reverberation model [dB]
att_max = 60.0 # Attenuation at the end of the simulation [dB]

beta = gpuRIR.beta_SabineEstimation(room_sz, T60) # Reflection coefficients
Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, T60) # Time to start the diffuse reverberation model [s]
Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)	 # Time to stop the simulation [s]
nb_img = gpuRIR.t2n( Tdiff, room_sz )	# Number of image sources in each dimension

print("coef : " + str(time.time()-tic)+ "sec")

tic = time.time()
RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_traj, pos_rcv, nb_img, Tmax, fs, Tdiff=Tdiff, orV_rcv=orV_rcv, mic_pattern=mic_pattern)
print("filter generated : " + str(time.time()-tic)+ "sec")

tic = time.time()
filtered_signal = gpuRIR.simulateTrajectory(source_signal, RIRs)
filtered_signal = filtered_signal/np.max(np.abs(filtered_signal))
print("filter applied : " + str(time.time()-tic)+ "sec" )

wavfile.write('filtered_signal.wav', fs, filtered_signal)
