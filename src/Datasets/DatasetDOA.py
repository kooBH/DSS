import os
from glob import glob
import torch
import librosa as rs
import numpy as np
import torch.nn.functional as F
import json
import feature

class DatasetDOA(torch.utils.data.Dataset) : 
    def __init__(self,hp,is_train=True):
        self.hp = hp
        if is_train : 
            root = hp.data.train
        else :
            root = hp.data.test

        self.list_data = glob(os.path.join(root,"*.wav"))
        self.list_data = list(filter(lambda k: not '_' in k.split('/')[-1], self.list_data))

    @staticmethod
    def get_feature(x,angle,pos_mic,hp) : 
        # [C, F, T] : complex
        spec = rs.stft(x,n_fft=hp.audio.n_fft)
        spec = torch.from_numpy(spec)

        # cut into shape of angles
        T = angle.shape[1]
        spec = spec[:,:,:T]

        list_feat = []

        # feature
        list_input = hp.model.input

        if "spec" in list_input : 
            list_feat.append(torch.cat((spec.real,spec.imag)))

        if "mag" in list_input : 
            list_feat.append(torch.abs(spec))

        if "LogPowerSpectral" in list_input : 
            list_feat.append(feature.LogPowerSpectral(spec))

        if "cosIPD" in list_input :
            list_feat.append(feature.cosIPD(spec))

        if "cossinIPD" in list_input:
            list_feat.append(feature.cossinIPD(spec))

        if "AF" in list_input : 
            list_feat.append(feature.AngleFeature(spec,angle,pos_mic))

        feat = torch.cat(list_feat)

        return feat.float()
        
    def __getitem__(self,idx):
        # /home/data2/kbh/DOA/v0_test/26.wav
        path = self.list_data[idx]

        tmp_split = self.list_data[idx].split("/")
        # 26.wav
        name_data = tmp_split[-1]
        # 26
        id_data = name_data.split(".")[0]
        # /home/data2/kbh/DOA/v0_test
        dir_data = "/".join(tmp_split[:-1])
        # /home/data2/kbh/DOA/v0_test/26.wav
        path_json = dir_data+"/"+id_data+".json"

        f_json = open(path_json)
        json_data = json.load(f_json)

        # meta data
        n_src = json_data["n_src"]
        pos_mic = torch.tensor(json_data["pos_mic"])
        azimuth = torch.tensor(json_data["azimuth"])
        elevation = torch.tensor(json_data["elevation"])

        # select target source
        idx_target = np.random.randint(n_src)
        azimuth = azimuth[idx_target]
        elevation = elevation[idx_target]
        angle = torch.stack((azimuth,elevation),-1)
        angle = torch.unsqueeze(angle,0)

        # clean
        path_clean = os.path.join(dir_data,id_data+"_"+str(idx_target)+".wav")
        clean,_ = rs.load(path_clean,sr=16000,mono=True)

        # load data
        raw,_ = rs.load(path,sr=16000,mono=False)


        feat = self.get_feature(raw,angle,pos_mic,self.hp)

        data = {"feat":feat,"clean":clean,"noisy":raw}

        return data

    def __len__(self):
        return len(self.list_data)


## DEV
if __name__ == "__main__" : 
    import sys
    sys.path.append("./")
    from utils.hparams import HParam
    hp = HParam("../config/SPEAR/v20.yaml","../config/SPEAR/default.yaml")
    db = DatasetSPEAR(hp,is_train=False)

    db[0]



    