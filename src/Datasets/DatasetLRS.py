import os
from glob import glob
import torch
import librosa as rs
import torch.nn.functional as F
import json
import numpy as np

class DatasetLRS(torch.utils.data.Dataset) : 
    def __init__(self,hp,is_train=True):
        self.hp = hp
        self.is_train = is_train
        if is_train : 
            self.root = hp.data.train
            self.root_vid = hp.data.video_train
        else :
            self.root = hp.data.test
            self.root_vid = hp.data.video_test
        self.list_data = glob(os.path.join(self.root,"noisy","*.wav"))
        print("DatasetLRS[train:{}] : {}".format(is_train,len(self.list_data)))

    def __getitem__(self,idx):
        path_noisy = self.list_data[idx]
        name = path_noisy.split("/")[-1]
        name = name.split(".")[0]
        path_label = os.path.join(self.root,"label","{}.json".format(name))

        noisy, _ = rs.load(path_noisy,sr=self.hp.audio.sr,mono=False)

        with open(path_label,"r") as f :
            label = json.load(f)

        # Select target
        n_src = len(label["angles"])

        idx_target = np.random.randint(n_src)

        path_clean = os.path.join(self.root,"clean","{}_{}.wav".format(name,idx_target))
        clean, _ = rs.load(path_clean,sr=self.hp.audio.sr)

        # Load Facial feature
        id_vid = label["id_videos"][idx_target]
        cls_vid = label["class_videos"][idx_target]
        n_frame = label["n_frames"][idx_target]
        path_vid = os.path.join(self.root_vid,cls_vid,id_vid+".npz")
        feat_face = torch.from_numpy(np.load(path_vid)["data"]).float()
        feat_face = feat_face[:n_frame,:,:] 
        
        
        feat_face = torch.permute(feat_face,(0,2,1))

        # T,W,D -> W,D,T to stretch
        #feat_face = torch.permute(feat_face,(1,2,0))

        # Stretch facial feature,
        # set all 0 to non-speech frames.
        if(label["idx_speech"][idx_target] == -1):
            idx_face = 0
            len_face = int((640/80000)*(label["len_speech"][idx_target]))
        else : 
            idx_face = int((640/80000)*label["idx_speech"][idx_target])
            len_face = int((640/80000)*(label["len_speech"][idx_target] - label["idx_speech"][idx_target]))

        plate_face = torch.zeros(1,512,640)
        #plate_face = torch.zeros(1,112,112,640)

        feat_face =  F.interpolate(feat_face,size=len_face, mode='nearest')

        plate_face[:,:,idx_face:idx_face+len_face] = feat_face[:,:,:]
        #plate_face[:,:,:,idx_face:idx_face+len_face] = feat_face[:,:,:]
        # C,W,D,T

        
        # plate_fate -> C,T,W,D
        #plate_face = torch.permute(plate_face,(0,3,1,2))

        # Label
        data={}
        data["noisy"] = torch.from_numpy(noisy).float()
        data["clean"] = torch.from_numpy(clean).float()
        data["angle"] = torch.tensor(label["angles"][idx_target]).float()
        data["mic_pos"] = torch.tensor(label["mic_pos"]).float()
        data["face"] = plate_face.float()

        if self.hp.data.shake_mic_pos and self.is_train:
            data["mic_pos"] = data["mic_pos"] + torch.rand_like(data["mic_pos"])*0.01

        return data

    def __len__(self):
        return len(self.list_data)

