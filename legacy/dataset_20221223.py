import os
from glob import glob
import torch
import numpy
import librosa
import numpy as np
import torch.nn.functional as F
import json
import feature
import pdb

## TODO 

def get_n_feature(
    n_channel,
    n_target,
    mono=False,
    phase="cossinIPD",
    full_phase = True
):
    # Angle Feature
    n_feature = 2*n_target

    # Mag 
    if mono :
        n_feature += 1
    else :
        n_feature += n_channel

    if full_phase :
        n_IPD = 6
    else :
        # NOTE : tmp
        n_IPD = 3

    if phase == "cossinIPD" :
        n_feature += 2*n_IPD
    else :
        n_feature += n_IPD

    return n_feature

"""
    audio : [n_sample,n_channel]
"""

def deemphasis(audio,preemphasis_coef=0.97,device='cuda:0'):
    if torch.is_tensor(audio) :
        de = torch.zeros(audio.shape).to(device)
    else :
        de = np.zeros(audio.shape)
    de[0,:] = audio[0,:]
    de[1,:] = audio[1,:]
    de[2,:] = audio[2,:]
    for i_sample in range(3,audio.shape[0]) : 
        de[i_sample] = audio[i_sample] + preemphasis_coef*de[i_sample-1]-preemphasis_coef*de[i_sample-2]+preemphasis_coef*de[i_sample-3]
    return de

def preprocess(
    raw,
    azimuth,
    elevation,
    pos_mic,
    phase_func,
    T=None,
    n_target=4,
    n_src=4,
    flat=True,
    mono=False,
    LPS=False,
    full_phase=True,
    only_azim = True,
    ADPIT = True
    ):
    ## input preprocessing
    # NOTE #1
    # [:T] due to 1 frame mismatch. Maybe because of shift?
    # need to check later.  

    if not torch.is_tensor(raw) : 
        raw = torch.from_numpy(raw)

    if T is not None : 
        stft = torch.stft(raw,n_fft=512,center=True,return_complex=True)[:,:,:T]
    else :
        stft = torch.stft(raw,n_fft=512,center=True,return_complex=True)
        T = stft.shape[2]

    if LPS : 
        mag_func = feature.LogPowerSpectral
    else :
        mag_func = feature.Mag

    if mono : 
        mag =  mag_func(stft[0:1,:,:])
    else :
        mag =  mag_func(stft[:,:,:])

    # Using selected phase preprocessor
    phase = phase_func(stft,full_phase)

    # Angle feature
    angle = torch.zeros((n_target,T,2))

    # trim
    azimuth = azimuth[:n_target,:T]
    elevation = elevation[:n_target,:T]

    if only_azim : 
        elevation[:,:] = 45
    # azimuth, elevation : [n_src,n_frame]
    angle[:n_src,:,:] =  torch.stack((azimuth,elevation),-1)

    dup = None
    ## dup for null target - angle
    if n_src < n_target:
        dup = numpy.random.choice(range(n_src),n_target-n_src)
        angle[n_src:n_target,:,:] = angle[dup,:,:]

    AF = feature.AngleFeature(stft,angle,pos_mic)

    if not ADPIT : 
        AF[n_src:,:,:] = 0 

    ## Flatten
    if flat : 
        # mag [C,F,T] -> [C*F,T]
        mag = torch.flatten(mag,end_dim=1)
        # phase [C-1,F,T] -> [(C-1)*F,T]
        phase = torch.flatten(phase,end_dim=1)
        # AF [N,F,T] -> [N*F,T]
        AF = torch.flatten(AF,end_dim=1) 
    # concat [F+F'+F'',T]
    input = torch.concat((mag,phase,AF))


    return input,stft,dup

def gen_DOA_SV(
    n_fft=512,
    n_direction=359,
    n_channel = 4,
    sound_speed = 340.3,
    sr=16000,
    dist = 100.0
    ):
    n_hfft = int(n_fft/2+1)

    ## Sensor map
    map_sensor = np.zeros((4,3))
    map_sensor[0,:]=[-0.04,-0.04,0.0]
    map_sensor[1,:]=[-0.04,0.04,0.0]
    map_sensor[2,:]=[0.04,-0.04,0.0]
    map_sensor[3,:]=[0.04,0.04,0.0]

    list_azim = np.zeros(n_direction)
    for i in range(n_direction):
        list_azim[i] = -180+i

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
    for i in range(n_direction) :
        for j in range(n_channel) :
            for k in range(n_hfft) :
                h[i,j,k] = np.exp(-1j*2*np.pi*k*(TDOA[i,j]-TDOA[i,j])*sr/n_fft)
    return torch.from_numpy(h)

class DatasetDOA(torch.utils.data.Dataset):
    def __init__(self,path,
    load_preprocessed=False,
    n_target = 4, 
    ADPIT=True,
    LPS=False,
    only_azim=True,
    azim_shaking=0,
    preemphasis=False,
    preemphasis_coef=0.97,
    preemphasis_order=3
    ):

        if load_preprocessed : 
            self.list_data = glob(os.path.join(path+"_pre","spec","*.pt"))
        else : 
            self.list_data = glob(os.path.join(path,"*.wav"))
            # filtering target audio
            self.list_data = list(filter(lambda k: not '_' in k.split('/')[-1], self.list_data))
        self.load_preprocessed = load_preprocessed
        self.path = path

        self.n_target = n_target

        self.preemphasis =preemphasis
        self.preemphasis_coef = preemphasis_coef
        self.preemphasis_order = preemphasis_order

        # deprecated 
        self.flat = True
        self.mono = False
        self.full_phase = True

        self.ADPIT = ADPIT
        self.LPS = LPS
        self.only_azim = only_azim
        self.azim_shaking=azim_shaking

        #self.SV = gen_DOA_SV()

    def __getitem__(self,idx):
        if self.load_preprocessed :
            return self.load_pt(idx)
        else :
            return self.load_wav(idx)


    def __len__(self):
        return len(self.list_data)

    def load_pt(self,idx):
        tmp_split = self.list_data[idx].split("/")
        name_data = tmp_split[-1]
        id_data = name_data.split(".")[0]

        dir_data = "/".join(tmp_split[:-1])
        path_json = self.path+"/"+id_data+".json"

        f_json = open(path_json)
        json_data = json.load(f_json)

        n_src = json_data["n_src"]
        # pos_mic 
        pos_mic = torch.tensor(json_data["pos_mic"])
        azimuth = torch.tensor(json_data["azimuth"])

        azimuth += (torch.rand(azimuth.shape)-0.5)*self.azim_shaking

        elevation = torch.tensor(json_data["elevation"])
        if self.only_azim : 
            elevation[:,:]=60
        T = elevation.shape[1]
        
        ## Loading input features
        if self.LPS : 
            LPS= torch.load(os.path.join(self.path+"_pre","LPS",name_data))
        else :
            LPS = torch.load(os.path.join(self.path+"_pre","mag",name_data))

        IPD = torch.load(os.path.join(self.path+"_pre","cossin_IPD",name_data))

        ## Loading raw data for criterion
        spec = torch.load(self.list_data[idx])
        spec = spec[:,:,:T,:]

        # to complex
        stft = spec[:,:,:,0] * spec[:,:,:,1]*1j

        # AngleFeature
        angle = torch.zeros((self.n_target,T,2))
        angle[:n_src,:,:] =  torch.stack((azimuth,elevation),-1)
        AF = feature.AngleFeature(stft,angle,pos_mic)
        if not self.ADPIT : 
            AF[n_src:,:,:] = 0 
        else :
            dup = numpy.random.choice(range(n_src),self.n_target-n_src)
            if n_src < self.n_target:
                AF[n_src:self.n_target,:,:] = AF[dup,:,:]

        LPS = torch.flatten(LPS[:,:,:T],end_dim=1)
        IPD = torch.flatten(IPD[:,:,:T],end_dim=1)
        AF  = torch.flatten(AF,end_dim=1)

        input = torch.cat((LPS,IPD,AF),dim=0)

        # Temporal treatment for 'Audio buffer is not finite everywhere' error
        try : 
            raw,_ = librosa.load(os.path.join(self.path,id_data+".wav"),sr=16000,mono=False)        
            # raw [n_channel, n_sample]
            t_raw = np.zeros(raw.shape)
            for i_sample in range(3,raw.shape[1]) :
                    t_raw[:,i_sample] = raw[:,i_sample] - self.preemphasis_coef*raw[:,i_sample-1] + self.preemphasis_coef * raw[:,i_sample-1] - self.preemphasis_coef * raw[:,i_sample-2]
            raw = t_raw
        except librosa.ParameterError as e:
            return self.__getitem__(idx+1)

        raw = torch.from_numpy(raw)
        ## load target routine
        
        ## target [N, C, T]
        target = torch.zeros(self.n_target, raw.shape[0] , raw.shape[1])
        for i  in range(n_src) : 
            tmp,_ = librosa.load(self.path+"/"+id_data+"_"+str(i)+".wav",sr=16000,mono=False)

            t_tmp = np.zeros(tmp.shape)
            for i_sample in range(3,tmp.shape[1]) :
                    t_tmp[:,i_sample] = tmp[:,i_sample] - self.preemphasis_coef*tmp[:,i_sample-1] + self.preemphasis_coef * tmp[:,i_sample-1] - self.preemphasis_coef * tmp[:,i_sample-2]
            tmp = t_tmp

            target[i,:,:] = torch.from_numpy(tmp)

        ## dup for null target - target wav
        if  n_src < self.n_target :
            if self.ADPIT : 
                target[n_src:self.n_target,:,:] = target[dup,:,:]
            else :
                target[n_src:self.n_target,:,:] = 0

        data = {"feature":input.float(),"spec":stft,"target":target.float(),"raw":raw[:,:].float(),"n_src":n_src}

        return data

    def load_wav(self,idx) : 
        tmp_split = self.list_data[idx].split("/")
        name_data = tmp_split[-1]
        id_data = name_data.split(".")[0]

        dir_data = "/".join(tmp_split[:-1])
        path_json = dir_data+"/"+id_data+".json"

        f_json = open(path_json)
        json_data = json.load(f_json)

        n_src = json_data["n_src"]
        # pos_mic 
        pos_mic = torch.tensor(json_data["pos_mic"])
        azimuth = torch.tensor(json_data["azimuth"])

        if n_src > self.n_target : 
            n_src = self.n_target

        if self.azim_shaking != 0 :
            azimuth += (torch.rand(azimuth.shape)-0.5)*self.azim_shaking

        elevation = torch.tensor(json_data["elevation"])

        T = elevation.shape[1]

        # Temporal treatment for 'Audio buffer is not finite everywhere' error
        try : 
            raw,_ = librosa.load(self.list_data[idx],sr=16000,mono=False)        
            # raw [n_channel, n_sample]
            if self.preemphasis : 
                t_raw = np.zeros(raw.shape)
                for i_sample in range(3,raw.shape[1]) :
                        t_raw[:,i_sample] = raw[:,i_sample] - self.preemphasis_coef*raw[:,i_sample-1] + self.preemphasis_coef * raw[:,i_sample-1] - self.preemphasis_coef * raw[:,i_sample-2]
                raw = t_raw
        except librosa.ParameterError as e:
            return self.__getitem__(idx+1)

        raw = torch.from_numpy(raw)

        input,stft,dup = preprocess(
            raw,
            azimuth,
            elevation,
            pos_mic,
            feature.cossinIPD,
            T=T,
            n_target=self.n_target,
            n_src=n_src,
            flat=self.flat,
            mono=self.mono,
            LPS=self.LPS,
            full_phase=self.full_phase,
            only_azim=self.only_azim,
            ADPIT=self.ADPIT)
        ## load target routine
        
        ## target [N, C, T]
        target = torch.zeros(self.n_target, raw.shape[0] , raw.shape[1])
        for i  in range(n_src) : 
            tmp,_ = librosa.load(dir_data+"/"+id_data+"_"+str(i)+".wav",sr=16000,mono=False)

            if self.preemphasis : 
                t_tmp = np.zeros(tmp.shape)
                for i_sample in range(3,tmp.shape[1]) :
                        t_tmp[:,i_sample] = tmp[:,i_sample] - self.preemphasis_coef*tmp[:,i_sample-1] + self.preemphasis_coef * tmp[:,i_sample-1] - self.preemphasis_coef * tmp[:,i_sample-2]
                tmp = t_tmp

            target[i,:,:] = torch.from_numpy(tmp)

        ## dup for null target - target wav
        if  n_src < self.n_target :
            if self.ADPIT : 
                target[n_src:self.n_target,:,:] = target[dup,:,:]
            else :
                target[n_src:self.n_target,:,:] = 0

        data = {"feature":input.float(),"spec":stft.cfloat(),"target":target.float(),"path_raw":self.list_data[idx],"raw":raw[:,:].float(),"n_src":n_src}
        return data


    """
        audio : [n_sample,n_channel]
    """
    def deemphasis(self,audio):
        de = np.zeros(audio.shape)
        de[0,:] = audio[0,:]
        de[1,:] = audio[1,:]
        de[2,:] = audio[2,:]
        for i_sample in range(3,audio.shape[0]) : 
            de[i_sample] = audio[i_sample] + self.preemphasis_coef*de[i_sample-1]-self.preemphasis_coef*de[i_sample-2]+self.preemphasis_coef*de[i_sample-3]

        return de

## Pre-process Data
if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    from multiprocessing import Pool, cpu_count

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in','-i' ,type=str, required=True)
    parser.add_argument('--dir_out','-o' ,type=str, required=True)
    args = parser.parse_args()

    dir_in = args.dir_in
    dir_out = args.dir_out


    list_target = [x for x in glob(os.path.join(dir_in,"*.wav"))]

    def filter_clean(x):
        name = x.split('/')[-1]
        if "_" in name :
            return False
        else :
            return True

    list_target = [x for x in list_target if filter_clean(x) ]

    def extract(idx):
        path = list_target[idx]
        name = path.split('/')[-1]
        id = name.split('.')[0]
        try : 
            raw,_ = librosa.load(path,sr=16000,mono=False)
        except librosa.ParameterError as e:
            print("ERROR on {} | {}".format(path,e))
            return

        t_raw = np.zeros(raw.shape)
        for i_sample in range(3,raw.shape[1]) :
                t_raw[:,i_sample] = raw[:,i_sample] - 0.97*raw[:,i_sample-1] + 0.97 * raw[:,i_sample-1] - 0.97 * raw[:,i_sample-2]
        raw = torch.from_numpy(t_raw)

        stft = torch.stft(raw,n_fft=512,center=True,return_complex=True)


        if extract_mag :
            mag = feature.Mag(stft[:,:,:])
            path_mag = os.path.join(dir_out,"mag",id +".pt")
            torch.save(mag.float(),path_mag)

        if extract_LPS : 
            lps = feature.LogPowerSpectral(stft[:,:,:])
            path_lps = os.path.join(dir_out,"LPS",id +".pt")
            torch.save(lps.float(),path_lps)

        if extract_ipd :
            ipd = feature.cossinIPD(stft[:,:,:],True)
            path_ipd = os.path.join(dir_out,"cossin_IPD",id+".pt")
            torch.save(ipd.float(),path_ipd)

        if extract_spec :
            re = stft.real
            im = stft.imag
            spec = torch.stack((re,im),-1)
            path_spec = os.path.join(dir_out,"spec",id +".pt")
            torch.save(spec.float(),path_spec)
    
    os.makedirs(os.path.join(dir_out,"mag"),exist_ok=True)
    os.makedirs(os.path.join(dir_out,"LPS"),exist_ok=True)
    os.makedirs(os.path.join(dir_out,"cossin_IPD"),exist_ok=True)
    os.makedirs(os.path.join(dir_out,"spec"),exist_ok=True)

    cpu_num = int(cpu_count()/4)

    extract_mag = False
    extract_LPS = False
    extract_ipd = True
    extract_spec = False

    os.makedirs(os.path.join(dir_out),exist_ok=True)
    arr = list(range(len(list_target)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(extract, arr), total=len(arr),ascii=True,desc='pre-processing {}'.format(dir_in)))