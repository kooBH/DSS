import sys,os
sys.path.append("gpuRIR")

from mixing import *
from glob import glob

import numpy as np
import librosa
import torch

import pandas as pd
import random

# utils
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

### PARAM ###
n_base = 100
n_train = n_base*10
n_test =  n_base*1
n_eval =  n_base*1
sec = 4
output_root = "/home/data2/kbh/DOA"
version="v1"
# Path
CHiME4_root = "/home/data/kbh/CHiME4/isolated_ext"

# ratio among total clean data
ratio_train = 0.9
ratio_dev = 0.05
ratio_eval = 0.05
split_top_db = 15

# ratio between speech and non-speech
ratio_speech = 0.5

###  END - PARAM ###
pool_mic = [
## Linear

# 2cm - X
[ [-0.03,0.00,0.00],
  [-0.01,0.00,0.00],
  [+0.01,0.00,0.00],
  [+0.03,0.00,0.00]],

# 3cm - X
[ [-0.045,0.00,0.00],
  [-0.015,0.00,0.00],
  [+0.015,0.00,0.00],
  [+0.045,0.00,0.00]],

# 4 cm - X
[ [-0.06,0.00,0.00],
  [-0.02,0.00,0.00],
  [+0.02,0.00,0.00],
  [+0.06,0.00,0.00]],

# 6 cm - X
[ [-0.09,0.00,0.00],
  [-0.03,0.00,0.00],
  [+0.03,0.00,0.00],
  [+0.09,0.00,0.00]],

# 2cm - Y
[ [0.00,-0.03,0.00],
  [0.00,-0.01,0.00],
  [0.00,+0.01,0.00],
  [0.00,+0.03,0.00]],

# 3cm - Y
[ [0.00,-0.045,0.00],
  [0.00,-0.015,0.00],
  [0.00,+0.015,0.00],
  [0.00,+0.045,0.00]],

# 4 cm - Y
[ [0.00,-0.06,0.00],
  [0.00,-0.02,0.00],
  [0.00,+0.02,0.00],
  [0.00,+0.06,0.00]],

# 6 cm - Y
[ [0.00,-0.09,0.00],
  [0.00,-0.03,0.00],
  [0.00,+0.03,0.00],
  [0.00,+0.09,0.00]],

## Square
# 2cm
[ [-0.01,-0.01,0.00],
  [-0.01,+0.01,0.00],
  [+0.01,-0.01,0.00],
  [+0.01,+0.01,0.00]],

# 4cm
[ [-0.02,-0.02,0.00],
  [-0.02,+0.02,0.00],
  [+0.02,-0.02,0.00],
  [+0.02,+0.02,0.00]],

# 6cm
[ [-0.03,-0.03,0.00],
  [-0.03,+0.03,0.00],
  [+0.03,-0.03,0.00],
  [+0.03,+0.03,0.00]],

# 8cm
[ [-0.04,-0.04,0.00],
  [-0.04,+0.04,0.00],
  [+0.04,-0.04,0.00],
  [+0.04,+0.04,0.00]],
]

if ratio_train + ratio_dev + ratio_eval != 1.0 : 
    raise Exception("sum of ratio is not 1.0")

list_speech = []

# CHiME
list_speech += [x for x in glob(os.path.join(CHiME4_root,"tr*","*.CH1.Clean.wav"))]

print("speech : {}".format(len(list_speech)))

# Split with ratio
## speech
np.random.shuffle(list_speech)

len_speech = len(list_speech)

end_speech_train = int(len_speech*ratio_train)
end_speech_dev = end_speech_train + int(len_speech*ratio_dev)

list_speech_train = list_speech[:end_speech_train]
list_speech_dev = list_speech[end_speech_train:end_speech_dev]
list_speech_eval = list_speech[end_speech_dev:]

print("speech-train : {}".format(len(list_speech_train)))
print("speech-dev: {}".format(len(list_speech_dev)))
print("speech-eval: {}".format(len(list_speech_eval)))

def gen_path(mode,is_speech=True):
    if mode == "train" and is_speech: 
        idx = np.random.randint(len(list_speech_train))
        path = list_speech_train[idx]
    elif mode == "dev" and is_speech: 
        idx = np.random.randint(len(list_speech_dev))
        path = list_speech_dev[idx]
    elif mode == "eval" and is_speech: 
        idx = np.random.randint(len(list_speech_eval))
        path = list_speech_eval[idx]
    else :
        raise Exception("ERROR::AIO::gen_path():unknown case {} {}".format(mode,is_speech))
    return path

def gen_v8(
    path_out,
    id_file,
    mode,
    min_src=2,
    max_src = 2,
    n_traj = 50,
    shift=128,
    max_SIR=10,
    max_RT60=0.9,
    fix = True,
    norm_signals=False,
    split_top_db = 25
    )->None:

    raws = []
    meta = {}
    len_max = 0
    len_min = 1e16

    sr=16000
    target_len = 16000*sec

    # mic array 
    pos_mic=pool_mic[np.random.randint(len(pool_mic))]
    random.shuffle(pos_mic)

    # shake array geometric
    pos_mic += np.random.rand(4,3)*0.01 

    meta["pos_mic"]=pos_mic

    # expand dim for matrix operation
    pos_mic = np.expand_dims(pos_mic,1)
    n_rec = len(pos_mic)

    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    #n_src = np.random.randint(low=min_src,high=5)
    n_src = max_src

    # load files 
    for path in range(n_src) :
        enough = False

        is_speech = True

        raw = None
        while not enough : 
            path_src = gen_path(mode,is_speech)
            temp, fs = librosa.load(path_src,mono=True,sr=sr)

            # norm
            temp = temp/np.max(np.abs(temp))

            # trim
            temp_split = librosa.effects.split(temp,split_top_db)    
            if not len(temp_split) > 0 :
                continue
            intv_chunk = temp_split[np.random.randint(len(temp_split))]
            temp = temp[intv_chunk[0]:intv_chunk[1]]

            #temp,_ = librosa.effects.trim(temp,top_db=10)

            # sample of sample
            len_temp = len(temp)
            if not is_speech:
                temp = temp[:np.random.randint(low=len_temp*0.5,high=len_temp)]

            # norm-chunk
            temp = temp/np.max(np.abs(temp))
            
            if raw is None : 
                raw = temp
            else :
                raw = np.concatenate((raw,temp))

            if len(raw) > target_len : 
                raw = raw[:target_len]
                enough = True

        raws.append(raw)

    meta["n_src"]=n_src

    # need to know RT60 to calculate audio length
    RT60 = np.random.uniform(low=0.1, high=max_RT60, size=None) 
    meta["RT60"]=RT60
    
    ## Matching length of sources
    padding=[]
  
    len_min = int(fs * sec - RT60*fs)  
    len_signals = len_min

    len_target = sec*fs

    # Debug : raws
    #for i in range(n_src):
    #    wavfile.write(path_out+"/tmp_"+str(i)+".wav", fs, raws[i])
    n_frame = int(np.ceil(len_target/shift))

    # generate room
    room = np.random.uniform(low=[5,5,2.5],high=[10,10,3.5])
    meta["room"]=room

    #s_idx = np.arange(n_src)
    #np.random.shuffle(s_idx)
#    s_idx = s_idx[:n_src]
    signals = []

    ### trajectory allocation ###
    traj_m,traj_s = gen_traj(room,n_src,fix=fix)
    traj_mm = np.tile(traj_m,(n_rec,1,1))

    traj_mm = traj_mm + pos_mic

    #print("traj_m : "+str(traj_m.shape))
    #print("traj_s : "+str(traj_s.shape))

    meta["traj_m"] = traj_m
    meta["traj_s"] = traj_s
    #meta["s_idx"] = s_idx

    SIRs = np.random.uniform(low=0, high=max_SIR, size=n_src)
    signal,signals,SIRs = mix(raws,SIRs,traj_s,traj_mm,room=room, RT60=RT60,norm_signals=norm_signals) 
    meta["SIRs"]=SIRs

    ## Save angles in label
    # match size of n_traj to n_frame for label
    traj_m_adj = np.zeros((n_frame,3))
    traj_adj = np.zeros((n_src,n_frame,3))    

    ratio = int(n_frame/n_traj)
    n_req_pad = n_frame - ratio*n_traj

    idx_adj = 0
    for i in range(n_traj):
        len_rep = ratio
        # padding
        if i < n_req_pad :
            len_rep +=1

        traj_m_adj[idx_adj:idx_adj+len_rep,:] = traj_m[i,:]
        traj_adj[:,idx_adj:idx_adj+len_rep:,:] = traj_s[:,i:i+1,:]
        idx_adj += len_rep

    azimuth = np.zeros((n_src,n_frame))
    elevation = np.zeros((n_src,n_frame))
    # Calculate Angle of Direct Path
    for i in range(n_src) : 
        for j in range(n_frame) : 
            dist = np.sqrt(np.power(traj_m_adj[j,0]-traj_adj[i,j,0],2) + np.power(traj_m_adj[j,1]-traj_adj[i,j,1],2))

            # azimuth
            tmp = np.arctan((traj_m_adj[j,1]-traj_adj[i,j,1])/(traj_m_adj[j,0]-traj_adj[i,j,0]))
            if traj_m_adj[j,0]  < traj_adj[i,j,0] : 
                azimuth[i,j] = 90 - np.degrees(tmp)
            else : 
                azimuth[i,j] = - 90 - np.degrees(tmp)
            # elevation
            tmp = np.arctan(dist/(traj_adj[i,j,2] - traj_m_adj[j,2] ))
            if traj_m_adj[j,2] < traj_adj[i,j,2] :
                elevation[i,j] = 90 - np.degrees(tmp)
            else :
                elevation[i,j] = - 90 - np.degrees(tmp)

    meta["azimuth"] = azimuth
    meta["elevation"]= elevation
    ## Match final audio output length
    if len(signal) < len_target :
        short = len_target - len(signal)
        signal = np.pad(signal,((0,short),(0,0)))
        for i in range(n_src) : 
            signals[i] = np.pad(signals[i],((0,short),(0,0)))
    elif len(signal) > len_target :
        over = len(signal) - len_target
        signal = signal[:-over]
        for i in range(n_src) : 
            signals[i] = signals[i][:-over]
    else :
        pass
     
    ## save
    wavfile.write(path_out+"/"+str(id_file)+".wav", fs, signal)
    for i in range(n_src):
        wavfile.write(path_out+"/"+str(id_file)+"_"+str(i)+".wav", fs, signals[i][:,:])

    with open(path_out+"/"+str(id_file)+".json", 'w') as f:
        json.dump(meta, f, indent=2,cls=NumpyEncoder)

def gen_train(idx):
    mode = "train"
    gen_v8(os.path.join(output_root,version+"_train"),
        idx,
        mode,
        min_src=1,
        n_traj = 50,
        shift=128,
        max_SIR=10,
        max_RT60=0.6,
        fix = True,
        norm_signals=False,
        split_top_db = split_top_db
        )

def gen_dev(idx):
    mode = "dev"
    gen_v8(os.path.join(output_root,version+"_test"),
        idx,
        mode,
        min_src=1,
        n_traj = 50,
        shift=128,
        max_SIR=10,
        max_RT60=0.6,
        fix = True,
        norm_signals=False,
        split_top_db = split_top_db
    )
if __name__=='__main__': 
    cpu_num = cpu_count()
    cpu_num = 32

    os.makedirs(os.path.join(output_root,version+"_train"),exist_ok=True)
    os.makedirs(os.path.join(output_root,version+"_test"),exist_ok=True)

    arr = list(range(n_test))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(gen_dev, arr), total=len(arr),ascii=True,desc='gen_test'))
    arr = list(range(n_train))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(gen_train, arr), total=len(arr),ascii=True,desc='gen_train'))
