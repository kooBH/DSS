import torch
import librosa
import soundfile as sf
import argparse

import numpy as np
from os.path import exists

import os,glob
from tqdm import tqdm

from cRFConvTasNet import cRFConvTasNet
from ptUtils.hparams import HParam
from dataset import preprocess,get_n_feature,deemphasis
import feature

from itertools import permutations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--chkpt',type=str,required=True)
    parser.add_argument('--dir_input','-i',type=str,required=True)
    parser.add_argument('--dir_output','-o',type=str,required=True)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    parser.add_argument('--n_frame','-n',type=int,required=False,default=125)
    parser.add_argument('--n_shift','-s',type=int,required=False,default=60)
    args = parser.parse_args()

    hp = HParam(args.config)
    print("NOTE::Loading configuration : "+args.config)
    n_target = hp.model.n_target
    preemphasis=hp.data.preemphasis

    device = args.device
    torch.cuda.set_device(device)

    N = n_target
    L_t = hp.model.l_filter_t
    L_f = hp.model.l_filter_f
    C = hp.data.n_channel
    F = int(hp.model.n_fft/2 + 1)
    shift = int(hp.model.n_fft/4)

    modelsave_path = args.chkpt
    os.makedirs(args.dir_output,exist_ok=True)

    n_feature = get_n_feature(
        hp.data.n_channel,
        hp.model.n_target, 
        mono=hp.model.mono, 
        phase = hp.model.phase,
        full_phase = hp.model.phase_full,
        )

    print("n_feature {}".format(n_feature))

    model = cRFConvTasNet(
        n_feature=n_feature,
        L_t=hp.model.l_filter_t,
        L_f=hp.model.l_filter_f,
        f_ch=hp.model.d_feature,
        n_fft=hp.model.n_fft,
        mask=hp.model.activation,
        n_target=n_target
    ).to(device)

    print("LOAD : {}".format(args.chkpt))
    model.load_state_dict(torch.load(args.chkpt, map_location=device))

    if hp.model.phase == "cossinIPD" : 
        phase_func = feature.cossinIPD
    else :
        phase_func = feature.cosIPD

    # mic array 
    pos_mic=[
        [-0.04,-0.04,0.00],
        [-0.04,+0.04,0.00],
        [+0.04,-0.04,0.00],
        [+0.04,+0.04,0.00]
    ]

    pos_mic = np.array(pos_mic)
    pos_mic = torch.from_numpy(pos_mic)

    list_target = [x for x in glob.glob(os.path.join(args.dir_input,"*.wav"))]

    preemphasis_coef = 0.97


    model.eval()
    with torch.no_grad():
        for path_target in tqdm(list_target) : 

            name_target = path_target.split('/')[-1]
            id_target = name_target.split('.')[0]

            raw,_ = librosa.load(path_target,sr=16000,mono=False)
           # print("{} {}".format(path_target,raw.shape))
            short = raw.shape[1]%128

            if short != 0 : 
                raw = raw[:,:-short]

            # preemphasis
            if True : 
                t_raw = np.zeros(raw.shape)
                for i_sample in range(3,raw.shape[1]) :
                        t_raw[:,i_sample] = raw[:,i_sample] -preemphasis_coef*raw[:,i_sample-1] + preemphasis_coef * raw[:,i_sample-1] - preemphasis_coef * raw[:,i_sample-2]
                raw = t_raw

           # print("raw : {} {}".format(raw.shape,short))

            raw = raw/np.max(np.abs(raw)).astype(np.float32)

            t_c = []
            t_c.append(raw[0,:])
            t_c.append(raw[1,:])
            t_c.append(raw[2,:])
            t_c.append(raw[3,:])

            """
            pos_mic
            +y
                2    4
                
                1    3    +x

            """

            raw[0,:] = t_c[1]
            raw[1,:] = t_c[0]

            # azimuth, elevation : [n_src,n_frame]

            # if there is .mat convert
            if exists(os.path.join(args.dir_input,id_target+".mat")) :
                # from mat bin
                import scipy.io
                label_mat = scipy.io.loadmat(os.path.join(args.dir_input, id_target + ".mat" ))

                # [n_frame, n_ch, 3]
                label_mat = label_mat["out_Data"].astype(np.int16)
                # TODO : Label conversion
                """
                    MATLAB n_shift : 256
                    DNN n_shift    : 128
                """
                azimuth = torch.from_numpy(np.repeat(label_mat[:,:,0],2,axis=0).T)
                azimuth = azimuth - 90
 
                #elevation = torch.from_numpy(np.repeat(label_mat[:,:,1],2,axis=0).T)
                #elevation = 90- elevation 

                elevation[:] = 45

            else :
                path_doa = os.path.join(args.dir_input,id_target+".npy")
                label = np.load(path_doa)
                azimuth   = torch.from_numpy(label[:,0,:])
                elevation = torch.from_numpy(label[:,1,:])
                elevation[:,:] = 30

            feature,stft,dup = preprocess(
                raw,
                azimuth,
                elevation,
                pos_mic,
                phase_func,
                mono=hp.model.mono,
                LPS=hp.model.LPS,
                full_phase=hp.model.phase_full,
                only_azim=hp.model.only_azim
                )
            # print("feature {}".format(feature.shape))
            T = stft.shape[2]
            feature = torch.unsqueeze(feature,dim=0)
            stft = torch.unsqueeze(stft,dim=0)

            feature = feature.to(device)

            filter =  model(feature.float())

            # Edge processing

            ## filtering
            # [B,C,F,T]
            input = stft.to(device)

            # dim of pad start from last dim
            input_alt = torch.nn.functional.pad(input,pad=(L_t,L_t,L_f,L_f) ,mode="constant", value=0)
            output = torch.zeros((input.shape[0],N,C,F,T),dtype=torch.cfloat).to(device)

            for t in range(2*L_t+1) : 
                for f in range(2*L_f+1):
                    for n in range(N) : 
                        output[:,n,:,:,:] += torch.mul(
                            input_alt[: , : , f:F+f , t:T+t ],
                            filter[:,n,:,f,t,:,:]
                            )

            # iSTFT
            output_raw = torch.zeros((1,N,C,raw.shape[1])).to(device)

            # torch does not supprot batch STFT/iSTFT
            for j in range(output_raw.shape[1]) :
                for k in range(output_raw.shape[2]) : 
                    output_raw[:,j,k,:] = torch.istft(output[:,j,k,:,:],n_fft = hp.model.n_fft)

            # Deemphasis
            #if hp.data.preemphasis :
            if True :
                for it_B in range(output_raw.shape[0]) :
                    for it_N in range(N) : 
                        output_raw[it_B,it_N] = (deemphasis(output_raw[it_B,it_N].T,device=device)).T

            output_raw = torch.sum(output_raw,dim=2)
            

            ## Normalization
            denom_max = torch.max(torch.abs(output_raw),dim=-1)[0]
            denom_max = torch.unsqueeze(denom_max,dim=-1)
            output_raw = output_raw/denom_max

            output_raw = output_raw.cpu().detach().numpy()

            for j in range(N) : 
                #sf.write(args.dir_output+'/'+id_target+'_'+str(j)+'.wav',output_raw[0,j,:,:].T,16000)
                #sf.write(args.dir_output+'/'+id_target+'_'+str(j)+'.wav',output_raw[0,j,0,:].T,16000)
                #sf.write(args.dir_output+'/'+id_target+'_'+str(j)+'.wav',output_raw[0,j,:,:].T,16000)
                sf.write(args.dir_output+'/'+id_target+'_'+str(j)+'.wav',output_raw[0,j,:].T,16000)




