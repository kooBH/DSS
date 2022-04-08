import os,glob
import torch
import librosa
import numpy as np
import torch.nn.functional as F
import json
import feature

class DatasetDOA(torch.utils.data.Dataset):
    def __init__(self,path,IPD="IPD"):
        self.list_data = glob.glob(os.path.join(path,"*.wav"))

        # filtering target audio
        self.list_data = list(filter(lambda k: not '_' in k.split('/')[-1], self.list_data))

        if IPD == "IPD" : 
            self.phase = feature.InterPhaseDifference
        elif IPD == "cosIPD" :
            self.phase = feature.cosIPD
        elif IPD == "sinIPD" :
            self.phase = feature.sinIPD
        elif IPD == "NIPD" : 
            self.phase = feautre.NormalizedIPD
        else : 
            raise Exception("Unimplemented phase method : {}".format(IPD))
        
    def __getitem__(self,idx):
        tmp_split = self.list_data[idx].split("/")
        name_data = tmp_split[-1]
        id_data = name_data.split(".")[0]

        dir_data = "/".join(tmp_split[:-1])
        path_json = dir_data+"/"+id_data+".json"

        f_json = open(path_json)
        json_data = json.load(f_json)

        n_src = json_data["n_src"]
        pos_mic = torch.tensor(json_data["pos_mic"])
        azimuth = torch.tensor(json_data["azimuth"])
        elevation = torch.tensor(json_data["elevation"])

        raw,_ = librosa.load(self.list_data[idx],sr=16000,mono=False)        

        raw = torch.from_numpy(raw)

        ## input preprocessing
        stft = torch.stft(raw,n_fft=1024)
        LPS =  feature.LogPowerSpectral(stft)

        # Using selected phase preprocessor
        phase = self.phase(stft)

        # Angle feature
        angle =  torch.stack((azimuth,elevation))
        print(angle.shape)
        AF = feature.AngleFeature(stft,angle,pos_mic)

        ## Flatten
        input = torch.flatten((LPS,phase,AF),start_dim=1)
        
        ## target
        target = torch.zeros(raw.shape)
        for i  in range(n_src) : 
            target[i,:] = librosa.load(dir_data+"/"+id_data+"_"+str(i)+".wav")

        data = {"input":input,"target":target}

        return data


    def __len__(self):
        return len(self.list_data)


if __name__ == "__main__":
    dataset = DatasetDOA(path="/home/data2/kbh/LGE/v2/",IPD="IPD")
    print(len(dataset))

    print(dataset[0])