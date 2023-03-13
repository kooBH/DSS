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

def AngleFeature(SV,stft):
    C,F,T = stft.shape
    ## split real,imag in AF only
    AF = torch.zeros(C,F,T, dtype=torch.cfloat)
    tmp_term =  SV[:,:,:]*(stft[:,:,:]/(stft[0:1,:,:]+1e-13))
    AF[:,:,:] = tmp_term/(torch.abs(tmp_term)+1e-13)
    AF = torch.sum(AF,axis=0)
    AF = torch.view_as_real(AF)                
    AF = torch.reshape(AF,(2,F,T))

    return AF

class DatasetDOA2(torch.utils.data.Dataset):
    def __init__(self,path,
    n_target = 1, 
    IPD="cossinIPD",
    # azimuth -> -0.5*azim_shaking ~ +0.5*azim_shaking
    azim_shaking=0,
    LPS=False
    ):
        if path[-1] == "/":
            path = path[:-1]
        self.path=path
        self.list_data = glob(os.path.join(path+"_pre","spec","*.pt"))
        print("DatasetDOA2 : {} from {}".format(len(self.list_data),path))
        self.LPS = LPS
        self.azim_shaking=azim_shaking

        self.SV = gen_DOA_SV()

    def __getitem__(self,idx):
        ## list_data[] : <path>_pre/stft/*.pt 
        tmp_split = self.list_data[idx].split("/")
        name_data = tmp_split[-1]
        id_data = name_data.split(".")[0]

        path_json = self.path+"/"+id_data+".json"

        f_json = open(path_json)
        json_data = json.load(f_json)

        # single random target from sources
        n_src = json_data["n_src"]
        idx_src = np.random.randint(n_src)

        # pos_mic 
        pos_mic = torch.tensor(json_data["pos_mic"])
        azimuth = torch.tensor(json_data["azimuth"][idx_src])

        if self.azim_shaking != 0 :
            azimuth += (torch.rand(azimuth.shape)-0.5)*self.azim_shaking

        # TODO azimuth fluctuation
        elevation = torch.tensor(json_data["elevation"][idx_src])

        T = elevation.shape[0]

        ## Load input feature
        spec = torch.load(self.list_data[idx])
        spec = spec[:,:,:T,:]

        # to complex
        stft = spec[:,:,:,0] * spec[:,:,:,1]*1j

        # stack on channel axis
        spec = torch.reshape(spec,(2*spec.shape[0],spec.shape[1],spec.shape[2]))

        ## gen AF
        angle = torch.zeros((1,T,2))
        angle[:,:,:] =  torch.stack((azimuth,elevation),-1)

        adj_angle = angle[0,:,0].type(torch.int16)+180
        adj_angle = torch.where(adj_angle < 359, adj_angle,adj_angle-359)

        SV = self.SV[adj_angle.tolist()]
        SV = torch.permute(SV,(1,2,0))

        AF = AngleFeature(SV,stft)

        input = torch.cat((spec,AF),dim=0)

        ## Load target 

        # pre-emphasis
        tmp,_ = librosa.load(self.path+"/"+id_data+"_"+str(idx_src)+".wav",sr=16000,mono=False)
        # 1ch target
        tmp = tmp[0:1,:]

        t_tmp = np.zeros(tmp.shape)
        for i_sample in range(3,tmp.shape[1]) :
                t_tmp[:,i_sample] = tmp[:,i_sample] - 0.97*tmp[:,i_sample-1] + 0.97 * tmp[:,i_sample-1] - 0.97 * tmp[:,i_sample-2]
        tmp = t_tmp

        target = torch.from_numpy(tmp)
        target = torch.stft(target,n_fft=512,center=True,return_complex=True)[:,:,:T]
        target = torch.cat((target.real,target.imag),dim=0)

        data = {"input":input,"target":target.float(),"id":id_data,"idx_src":idx_src}

        return data

    def __len__(self):
        return len(self.list_data)
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
def routine_extract():
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
            ipd = feature.cossinIPD(stft[:,:,:])
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

    cpu_num = cpu_count()

    extract_mag = True
    extract_LPS = True
    extract_ipd = True
    extract_spec = True

    os.makedirs(os.path.join(dir_out),exist_ok=True)
    arr = list(range(len(list_target)))
    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(extract, arr), total=len(arr),ascii=True,desc='pre-processing {}'.format(dir_in)))


def routine_dataset_test():
    dataset = DatasetDOA2("/home/data2/kbh/LGE/v9_AIO_test/",azim_shaking=10)

    data = dataset[0]
    print(data["input"].shape)
    print(data["input"].type())
    print(data["target"].shape)

    print("=================")
    mean = torch.mean(torch.abs(data["input"]),dim=1)
    mean = torch.mean(mean,dim=1)
    print(mean)


if __name__ == "__main__":

    #routine_extract()
    routine_dataset_test()
    