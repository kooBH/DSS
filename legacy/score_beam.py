import librosa
import torch
import soundfile as sf
import argparse
import os
import numpy as np
import json
from ptUtils.metric import SIR,PESQ

from tqdm import tqdm

from dataset import DatasetDOA

from ptUtils.hparams import HParam

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_est','-i',type=str,required=True)
    parser.add_argument('--dir_ref','-r',type=str,required=True)
    parser.add_argument('--n_data','-n',type=int,required=True)
    args = parser.parse_args()

    n_data = args.n_data

    SIR_eval = torch.zeros(4)
    PESQ_eval = torch.zeros(4)
    cnt_PESQ = [0,0,0,0]
    cnt_SIR = [0,0,0,0]

    for idx in tqdm(range(n_data)) : 
        f = open(args.dir_ref+"/"+str(idx)+".json",'r')
        j = json.load(f)
        n_src = j["n_src"]

        est = []
        ref = []
        for n in range(n_src): 
            path_est = os.path.join(args.dir_est,str(idx)+"_"+str(n)+".wav")
            path_ref = os.path.join(args.dir_ref,str(idx)+"_"+str(n)+".wav")
            t_est,_ = librosa.load(path_est,sr=16000,mono=True)
            t_ref,_ = librosa.load(path_ref,sr=16000,mono=True)

            est.append(t_est)
            ref.append(t_ref)

        est = torch.from_numpy(np.array(est))
        ref = torch.from_numpy(np.array(ref))

        if n_src != 1 : 
            SIR_eval[n_src-1] += SIR(est, ref ,device="cpu")
            cnt_SIR[n_src-1] +=1

        for n in range(n_src)  :
            PESQ_eval[n_src-1] += PESQ(est[n,:],ref[n,:])
            cnt_PESQ[n_src-1] +=1

    print("SIR {} | PESQ {}".format(
        torch.sum(SIR_eval[1:])/np.sum(cnt_SIR[1:]),
        torch.sum(PESQ_eval)/np.sum(cnt_PESQ)
    ))

    for i in range(4) : 
        SIR_eval[i] /=cnt_SIR[i]
        PESQ_eval[i] /=cnt_PESQ[i]
        if i != 0 : 
            print("n_src :  {} | SIR : {} | PESQ : {}".format(i+1,SIR_eval[i],PESQ_eval[i]))
        else :
            print("n_src :  {} | SIR : -- | PESQ : {}".format(i+1,PESQ_eval[i]))

       



        

