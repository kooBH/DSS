import os,glob
import torch
import numpy
import librosa
import numpy as np
import torch.nn.functional as F
import json
import feature

import pdb

class DatasetDOA(torch.utils.data.Dataset):
    def __init__(self,path,n_target = 4, IPD="cosIPD"):
        self.list_data = glob.glob(os.path.join(path,"*.wav"))
        # filtering target audio
        self.list_data = list(filter(lambda k: not '_' in k.split('/')[-1], self.list_data))

        self.n_target = n_target

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
        # pos_mic 
        pos_mic = torch.tensor(json_data["pos_mic"])
        azimuth = torch.tensor(json_data["azimuth"])
        elevation = torch.tensor(json_data["elevation"])

        T = elevation.shape[1]

        # Temporal treatment for 'Audio buffer is not finite everywhere' error
        try : 
            raw,_ = librosa.load(self.list_data[idx],sr=16000,mono=False)        
        except librosa.ParameterError as e:
            return self.__getitem__(idx+1)

        raw = torch.from_numpy(raw)

        ## input preprocessing
        # NOTE #1
        # [:T] due to 1 frame mismatch. Maybe because of shift?
        # need to check later.  
        stft = torch.stft(raw,n_fft=512,center=True,return_complex=True)[:,:,:T]
        LPS =  feature.LogPowerSpectral(stft[0:1,:,:])

        # Using selected phase preprocessor
        phase = self.phase(stft)

        # Angle feature
        angle = torch.zeros((self.n_target,T,2))
        angle[:n_src,:,:] =  torch.stack((azimuth,elevation),-1)

        ## dup for null target - angle
        dup = numpy.random.choice(range(n_src),self.n_target-n_src)
        if n_src < self.n_target:
            angle[n_src:self.n_target,:,:] = angle[dup,:,:]

        AF = feature.AngleFeature(stft,angle,pos_mic)

        ## Flatten
        # LPS [1,F,T] -> [F,T]
        LPS = torch.flatten(LPS,end_dim=1)
        # phase [C-1,F,T] -> [(C-1)*F,T]
        phase = torch.flatten(phase,end_dim=1)
        # AF [N,F,T] -> [N*F,T]
        AF = torch.flatten(AF,end_dim=1) 

        # concat [F+F'+F'',T]
        input = torch.concat((LPS,phase,AF))
        
        ## target [N, C, T]
        target = torch.zeros(self.n_target, raw.shape[0] , raw.shape[1])
        for i  in range(n_src) : 
            tmp,_ = librosa.load(dir_data+"/"+id_data+"_"+str(i)+".wav",sr=16000,mono=False)
            target[i,:,:] = torch.from_numpy(tmp)

        ## dup for null target - target wav
        if  n_src < self.n_target :
              target[n_src:self.n_target,:,:] = target[dup,:,:]


        data = {"flat":input,"spec":stft,"target":target,"path_raw":self.list_data[idx],"raw":raw[:,:]}

        return data


    def __len__(self):
        return len(self.list_data)


if __name__ == "__main__":
    dataset = DatasetDOA(path="/home/data2/kbh/LGE/v2/",IPD="IPD")
    print(len(dataset))

    for i in range(10) :
        print(i)
        print(dataset[i]["flat"].shape)
        print(dataset[i]["spec"].shape)
        print(dataset[i]["target"].shape)
