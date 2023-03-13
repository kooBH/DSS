import os
from glob import glob
import torch
import librosa as rs
import torch.nn.functional as F
import json

class DatasetMIDR(torch.utils.data.Dataset) : 
    def __init__(self,hp,is_train=True):
        self.hp = hp
        if is_train : 
            self.root = hp.data.train
        else :
            self.root = hp.data.test
        self.list_data = glob(os.path.join(self.root,"noisy","*.wav"))
        print("DatasetMIDR[train:{}] : {}".format(is_train,len(self.list_data)))
        
    def __getitem__(self,idx):

        path_noisy = self.list_data[idx]
        name = path_noisy.split("/")[-1]
        name = name.split(".")[0]
        path_clean = os.path.join(self.root,"clean","{}.wav".format(name))
        path_label = os.path.join(self.root,"label","{}.json".format(name))

        noisy, _ = rs.load(path_noisy,sr=self.hp.audio.sr,mono=False)
        clean, _ = rs.load(path_clean,sr=self.hp.audio.sr)

        with open(path_label,"r") as f :
            label = json.load(f)

        gap = float(label["mic_array"][0])*0.01

        mic_pos = torch.zeros(4,3)

        for i in range(4):
            mic_pos[i,1] = (-2 +i)*gap

        data={}
        data["noisy"] = torch.from_numpy(noisy).float()
        data["clean"] = torch.from_numpy(clean).float()
        data["angle"] = torch.tensor(label["angle"]).float()
        data["mic_pos"] = mic_pos.float()

        return data

    def __len__(self):
        return len(self.list_data)

