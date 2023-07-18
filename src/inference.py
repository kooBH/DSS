import torch
import librosa as rs
import soundfile as sf
import argparse
import os,glob
import numpy as np
import json

from tqdm.auto import tqdm

from ptUtils.hparams import HParam

from common import run,get_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,help="yaml for configuration")
    parser.add_argument('--default', type=str, default=None,help="default configuration")
    parser.add_argument('--chkpt',type=str,required=True)
    parser.add_argument('--dir_input','-i',type=str,required=True)
    parser.add_argument('--dir_output','-o',type=str,required=True)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    args = parser.parse_args()


    hp = HParam(args.config,args.default)
    print("NOTE::Loading configuration : "+args.config)

    device = args.device
    torch.cuda.set_device(device)

    batch_size = 1
    num_workers = 1

    modelsave_path = args.chkpt
    os.makedirs(args.dir_output,exist_ok=True)

    model = get_model(hp).to(device)
    print("NOTE::Loading pre-trained model : "+ args.chkpt)
    model.load_state_dict(torch.load(args.chkpt, map_location=device))

    list_input = glob.glob(os.path.join(args.dir_input,'*.wav'))

    #### EVAL ####
    model.eval()
    with torch.no_grad():
        for path in tqdm(list_input) : 
            noisy ,_ = rs.load(path,sr=hp.audio.sr,mono=False)
            noisy = torch.from_numpy(noisy)
            noisy = torch.unsqueeze(noisy,0).to(device)
            name_target = path.split("/")[-1]
            id_target = name_target.split(".")[0]

            path_json = os.path.join(args.dir_input,id_target+'.json')

            f_label = open(path_json,'r')
            json_data = json.load(f_label)
            f_label.close()

            # meta data
            #n_src = json_data["n_src"]
            n_src = 2
            mic_pos = torch.tensor(json_data["mic_pos"])
            mic_pos = torch.unsqueeze(mic_pos,0).to(device)

            angles = torch.tensor(json_data["angles"])

            # select target source
            for idx_target in range(n_src) : 
                i_angle= angles[idx_target]
                i_angle = torch.unsqueeze(i_angle,0).to(device)

                estim = model(noisy,i_angle,mic_pos)

                estim = estim[0].cpu().numpy()
                sf.write(os.path.join(args.dir_output,'{}_{}.wav'.format(id_target,idx_target)),estim,hp.audio.sr)



