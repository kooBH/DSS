import os,glob
import scipy
import torch
import numpy as np
import librosa as rs
import soundfile as sf
from scipy import io
import random
import json

## PATH
root_rir    = "/home/data2/kbh/MIDR___Multi-Channel_Impulse_Response_Database/"
root_speech = "/home/nas/DB/CHiME4/data/audio/16kHz/isolated/tr05_org/"
root_noise  = "/home/nas/DB/DEMAND/dataset/"

root_output = "/home/data2/kbh/DSS/MIDR/v0"

## PARAM
SNR_max = 20
SNR_min = 5

SIR_max = 10
SIR_min = 0

scale_dB_max = 15
scale_dB_min = -40

len_data = 16000*5
ratio_test = 0.25

## list-up
list_room_train = []
list_room_train.append("Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.160s)_3-3-3-8-3-3-3")
list_room_train.append("Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.360s)_3-3-3-8-3-3-3")
list_room_train.append("Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.610s)_3-3-3-8-3-3-3")
list_room_train.append("Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.160s)_8-8-8-8-8-8-8")
list_room_train.append("Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.360s)_8-8-8-8-8-8-8")
list_room_train.append("Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.610s)_8-8-8-8-8-8-8")

list_room_test = []
list_room_test.append("Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.160s)_4-4-4-8-4-4-4")
list_room_test.append("Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.360s)_4-4-4-8-4-4-4")
list_room_test.append("Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.610s)_4-4-4-8-4-4-4")

print("RIR train : {} | RIR test : {}".format(len(list_room_train),len(list_room_test)))

list_speech = glob.glob(os.path.join(root_speech,"*.wav"))
random.shuffle(list_speech)
idx_split = int(len(list_speech)*ratio_test)
list_speech_train = list_speech[idx_split:]
list_speech_test = list_speech[:idx_split]
print("speech train : {} | speech test : {}".format(len(list_speech_train),len(list_speech_test)))

list_noise = glob.glob(os.path.join(root_noise,"**","*.wav"))
random.shuffle(list_noise)
idx_split = int(len(list_noise)*ratio_test)
list_noise_train = list_noise[idx_split:]
list_noise_test = list_noise[:idx_split]
print("noise train : {} | noise test : {}".format(len(list_noise_train),len(list_noise_test)))

def sample(is_train=True) : 
    
    if is_train : 
        dir_RIR = random.sample(list_room_train,1)[0]
    else :
        dir_RIR = random.sample(list_room_test,1)[0]

    RT60 = float(dir_RIR[65:70])
    mic_array = dir_RIR[73:86]
    mic_on_left = bool(random.getrandbits(1))

    list_RIR = glob.glob(os.path.join(root_rir,dir_RIR,"*.mat"))

    is_2m = bool(random.getrandbits(1))

    if is_2m : 
        list_RIR = [x for x in list_RIR if "2m" in x.split('/')[-1] ]
    else : 
        list_RIR = [x for x in list_RIR if "1m" in x.split('/')[-1] ]

    idx_angle = np.random.choice(13,2,replace=False)

    RIR_speech = list_RIR[idx_angle[0]]
    RIR_interf = list_RIR[idx_angle[1]]

    angle = RIR_speech.split("_")[-1]
    angle = float(angle.split(".")[0])

    ## Scale

    SNR = np.random.uniform(low=SNR_min, high=SNR_max)
    SIR = np.random.uniform(low=SIR_min, high=SIR_max)
    scale_dB = np.random.uniform(low=scale_dB_min, high=scale_dB_max)

    ## audio
    if is_train :
        idx_speech = np.random.choice(len(list_speech_train),2,replace=False)
        path_speech = list_speech_train[idx_speech[0]]
        path_interf = list_speech_train[idx_speech[1]]
        path_noise = random.sample(list_noise_train,1)[0]
    else :
        idx_speech = np.random.choice(len(list_speech_test),2,replace=False)
        path_speech = list_speech_test[idx_speech[0]]
        path_interf = list_speech_test[idx_speech[1]]
        path_noise = random.sample(list_noise_test,1)[0]

    label = {}
    label["RT60"] = RT60
    label["mic_array"] = mic_array
    label["mic_on_left"] = mic_on_left
    label["is_2m"] = is_2m
    label["RIR_speech"] = RIR_speech
    label["RIR_interf"] = RIR_interf
    label["SIR"] = SIR
    label["SNR"] = SNR
    label["scale_dB"] = scale_dB
    label["angle"] = angle
    label["path_speech"] = path_speech
    label["path_interf"] = path_interf
    label["path_noise"] = path_noise

    return label, list_RIR

def mix(
        SIR,
        SNR,
        len_data,
        scale_dB, 
        mic_on_left,
        path_RIR_speech,
        path_RIR_interf,
        path_RIR_noise,
        path_speech,
        path_interf,
        path_noise
    ) :

    ## RIR and target
    rir_speech = io.loadmat(path_RIR_speech)["impulse_response"]
    if mic_on_left : 
        rir_speech = rir_speech[:,:4]
    else :
        rir_speech = rir_speech[:,4:]
    speech,_ = rs.load(path_speech,sr=16000,mono=False)

    s = []
    for i in range(4) :
        #s.append(scipy.signal.fftconvolve(speech,rir_speech[:,i]))
        s.append(scipy.signal.convolve(speech,rir_speech[:,i]))

    s = np.stack(s)    
    if s.shape[1] < len_data : 
        s = np.pad(s,(0,len_data-s.shape[1]))
    else :
        idx_clip = np.random.randint(0,s.shape[1]-len_data)
        s = s[:,:len_data]

    clean = s[0]
    clean_rms = (clean ** 2).mean() ** 0.5

    ## interf
    rir_interf = io.loadmat(path_RIR_interf)["impulse_response"]
    if mic_on_left : 
        rir_interf = rir_interf[:,:4]
    else :
        rir_interf = rir_interf[:,4:]
    interf,_ = rs.load(path_interf,sr=16000,mono=False)

    v = []
    for i in range(4) :
        #v.append(scipy.signal.fftconvolve(interf,rir_interf[:,i]))
        v.append(scipy.signal.convolve(interf,rir_interf[:,i]))
    v = np.stack(v)

    if v.shape[1] < len_data : 
        v = np.pad(v,(0,len_data-v.shape[1]))
    else :
        v = v[:,:len_data]

    interf = v[0]
    # SIR

    interf_rms = (interf ** 2).mean() ** 0.5
    snr_scalar = clean_rms / (10 ** (SIR / 20)) / (interf_rms + 1e-13)
    v *= snr_scalar

    ## noise
    noise = rs.load(path_noise,sr=16000)[0]
    # sample noise
    idx_clip = np.random.randint(0,noise.shape[0]-len_data)
    noise  = noise[idx_clip:idx_clip+len_data]

    # for all directions
    d = None
    for i in range(len(path_RIR_noise)) :
        n = []
        rir_noise = io.loadmat(path_RIR_noise[i])["impulse_response"]
        if mic_on_left : 
            rir_noise = rir_noise[:,:4]
        else :
            rir_noise = rir_noise[:,4:]

        for j in range(4) : 
            #n.append(scipy.signal.fftconvolve(noise,rir_noise[:,j]))
            n.append(scipy.signal.convolve(noise,rir_noise[:,j]))
        n = np.stack(n)
        
        if n.shape[1] < len_data : 
            n = np.pad(n,(0,len_data-n.shape[1]))
        else :
            n = n[:,:len_data]
        
        if d is None : 
            d = n
        else :
            d += n
    noise = d[0]
    noise_rms = (noise ** 2).mean() ** 0.5
    snr_scalar = noise_rms / (10 ** (SNR / 20)) / (noise_rms + 1e-13)
    d *= snr_scalar

    ## Mix
    x = s + v + d

    # dB Management
    # resacle noisy RMS
    rms = np.sqrt(np.mean(x ** 2))
    scalar = 10 ** (scale_dB / 20) / (rms + 1e-13)
    x *= scalar
    clean *= scalar

    if np.any(np.abs(x) > 0.999)  : 
        noisy_scalar = np.max(np.abs(x)) / (0.99 - 1e-13)  # same as divide by 1
        x /= noisy_scalar
        clean /= noisy_scalar

    return x, clean


def generate(idx,is_train) : 
    label,RIR_diffuse = sample(is_train)

    x,s = mix(
        SIR = label["SIR"],
        SNR     = label["SNR"],
        len_data = len_data,
        scale_dB = label["scale_dB"], 
        mic_on_left = label["mic_on_left"],
        path_RIR_speech = label["RIR_speech"],
        path_RIR_interf =  label["RIR_interf"],
        path_RIR_noise = RIR_diffuse,
        path_speech = label["path_speech"],
        path_interf = label["path_interf"],
        path_noise = label["path_noise"]
    )

    if is_train :
        train_test = "train"
    else :
        train_test = "test"


    path_x = os.path.join(root_output,train_test,"noisy","{}.wav".format(idx))
    path_s = os.path.join(root_output,train_test,"clean","{}.wav".format(idx))
    path_j = os.path.join(root_output,train_test,"label","{}.json".format(idx))

    sf.write(path_x,x.T,16000)
    sf.write(path_s,s.T,16000)
    with open(path_j,"w") as f :
        json.dump(label,f)

def generate_train(idx) : 
    generate(idx,is_train=True)

def generate_test(idx) : 
    generate(idx,is_train=False)

if __name__ == "__main__" : 
    # utils
    from tqdm.auto import tqdm
    from multiprocessing import Pool, cpu_count

    # Due to 'PySoundFile failed. Trying audioread instead' 
    import warnings
    warnings.filterwarnings('ignore')

    cpu_num = int(cpu_count()/2)

    for c1 in ["train","test"] :
        for c2 in ["clean","noisy","label"] :
            os.makedirs(os.path.join(root_output,c1,c2),exist_ok=True)

    arr = list(range(3000))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(generate_train, arr), total=len(arr),ascii=True,desc='processing'))

    arr = list(range(1000))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(generate_test, arr), total=len(arr),ascii=True,desc='processing'))