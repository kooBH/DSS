import torch
import librosa as rs
import soundfile as sf
import argparse
import os,glob
import numpy as np
import json

from tqdm.auto import tqdm

from ptUtils.hparams import HParam
from Datasets.DatasetUDSS import DatasetUDSS
from ptUtils.metric import run_metric

from common import run,get_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,help="yaml for configuration")
    parser.add_argument('--default', type=str, default=None,help="default configuration")
    parser.add_argument('--chkpt',type=str,required=True)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    parser.add_argument('--dir_output',type=str,required=True)
    args = parser.parse_args()

    hp = HParam(args.config,args.default)
    print("NOTE::Loading configuration : "+args.config)

    device = args.device
    torch.cuda.set_device(device)

    batch_size = 1
    num_workers = 1

    modelsave_path = args.chkpt
    os.makedirs(args.dir_output,exist_ok=True)


    root = hp.data.test
    list_input = glob.glob(os.path.join(root,"noisy","*.wav"))
    print(len(list_input))

    model = get_model(hp).to(device)
    print("NOTE::Loading pre-trained model : "+ args.chkpt)
    model.load_state_dict(torch.load(args.chkpt, map_location=device))

    #### EVAL ####
    model.eval()
    with torch.no_grad():
        f = open(args.dir_output+"/log.csv",'w')
        f.write("ID, anlges, PESQ, STIO, n_src, SIR, SNR, scale_dB\n")
        for idx,path in tqdm(enumerate(list_input)) : 
            name = path.split("/")[-1]
            name = name.split(".")[0]

            noisy ,_ = rs.load(path,sr=hp.audio.sr,mono=False)
            noisy = torch.from_numpy(noisy)
            noisy = torch.unsqueeze(noisy,0).to(device)
            name_target = path.split("/")[-1]
            id_target = name_target.split(".")[0]

            path_json = os.path.join(root,"label",id_target+'.json')


            with open(path_json,"r") as file_json :
                json_data = json.load(file_json)

            # meta data
            n_src = json_data["n_src"]
            mic_pos = torch.tensor(json_data["mic_pos"])
            mic_pos = torch.unsqueeze(mic_pos,0).to(device)
            angles = torch.tensor(json_data["angles"])

            PESQ = 0.0
            STOI = 0.0

            # select target source
            for idx_target in range(n_src) : 
                i_angle= angles[idx_target]
                i_angle = torch.unsqueeze(i_angle,0).to(device)

                estim = model(noisy,i_angle,mic_pos)
                estim = estim[0].cpu().numpy()


                path_clean = os.path.join(root,"clean","{}_{}.wav".format(name,idx_target))
                clean, _ = rs.load(path_clean,sr=hp.audio.sr)

                sf.write(os.path.join(args.dir_output,'{}_{}.wav'.format(id_target,idx_target)),estim,hp.audio.sr)

                PESQ += run_metric(estim,clean,'PESQ')
                STOI += run_metric(estim,clean,'STOI')
            
            PESQ /= n_src
            STOI /= n_src

            # log
            #f.write("ID, anlges, PESQ, STIO, n_src, SIR, SNR, scale_dB\n")
            if "SNR" in json_data :
                f.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(idx,json_data["angles"],PESQ,STOI, n_src,json_data["SIR"],json_data["SNR"],json_data["scale_dB"]))
            else :
                f.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(idx,json_data["angles"],PESQ,STOI, n_src,json_data["SIR"],"-",json_data["scale_dB"]))

        f.close()




