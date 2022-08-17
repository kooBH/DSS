import os
import torch
import numpy
import numpy as np
import torch.nn.functional as F
import pdb

## Pre-process Data
if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    from multiprocessing import Pool, cpu_count

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_out','-o' ,type=str, required=True)
    args = parser.parse_args()

    dir_out = args.dir_out

    n_fft = 512
    n_hfft = int(n_fft/2+1)
    n_direction = 359
    n_channel = 4
    sound_speed = 340.3
    sr=16000
    dist = 100.0

    ## Sensor map
    map_sensor = np.zeros((4,3))
    map_sensor[0,:]=[-0.04,-0.04,0.0]
    map_sensor[1,:]=[-0.04,0.04,0.0]
    map_sensor[2,:]=[0.04,-0.04,0.0]
    map_sensor[3,:]=[0.04,0.04,0.0]

    list_azim = np.zeros(n_direction)
    for i in range(n_direction):
        list_azim[i] = i

    map_source = np.zeros((n_direction,3))
    for i in range(n_direction):
        map_source[i,:] = [ dist*np.cos(np.deg2rad(90-list_azim[i])),dist*np.sin(np.deg2rad(90-list_azim[i])),0 ]    
    #print(map_source)

    # Calculate TDOA vector
    TDOA = np.zeros((n_direction, n_channel))
    for i in range(n_direction):
        for j in range(n_channel):
            pdist = np.linalg.norm(map_sensor[j,:] - map_source[i])
            TDOA[i,j] = pdist/sound_speed

    # Estimate RIR
    h = np.zeros((n_direction, n_channel, n_hfft),np.cfloat)
    for i in tqdm(range(n_direction)) :
        for j in range(n_channel) :
            for k in range(n_hfft) :
                h[i,j,k] = np.exp(-1j*2*np.pi*k*(TDOA[i,j]-TDOA[i,j])*sr/n_fft)

    os.makedirs(os.path.join(dir_out),exist_ok=True)
    torch.save(h,dir_out+"/filter.pt")