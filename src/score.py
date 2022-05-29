import torch
import librosa
import soundfile as sf
import argparse
import os
import numpy as np
from ptUtils.metric import SIR,PESQ

from tqdm import tqdm

from dataset import DatasetDOA
from cRFConvTasNet import cRFConvTasNet

from ptUtils.hparams import HParam


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--chkpt',type=str,required=True)
    parser.add_argument('--dir_input','-i',type=str,required=True)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    args = parser.parse_args()

    hp = HParam(args.config)
    print("NOTE::Loading configuration : "+args.config)
    n_target = hp.model.n_target

    device = args.device
    torch.cuda.set_device(device)

    batch_size = 1
    num_workers = 1

    N = n_target
    L = hp.model.l_filter
    C = hp.data.n_channel
    F = int(hp.model.n_fft/2 + 1)
    ## TODO : have to be availiable to arbitary length of audio
    T = hp.data.n_frame
    shift = int(hp.model.n_fft/4)

    modelsave_path = args.chkpt

    dataset = DatasetDOA(args.dir_input,n_target)
    loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,num_workers=num_workers)

    model = cRFConvTasNet(
        L=hp.model.l_filter,
        f_ch=hp.model.d_feature,
        n_fft=hp.model.n_fft,
        mask=hp.model.activation,
        n_target=n_target
    ).to(device)

    model.load_state_dict(torch.load(args.chkpt, map_location=device))

    #### EVAL ####
    model.eval()
    SIR_eval = torch.zeros(4).to(device)
    PESQ_eval = torch.zeros(4).to(device)
    cnt_PESQ = [0,0,0,0]
    cnt_SIR = [0,0,0,0]
    with torch.no_grad():
        test_loss =0.   
        for i, (batch_data) in tqdm(enumerate(loader),total=len(dataset)):
            # run model
            feature = batch_data['flat'].to(device)
            # output = [B,C, 2*L+1,2*L+1, n_hfft, Time]
            filter = model(feature)

            ## filtering
            # [B,C,F,T]
            input = batch_data['spec'].to(device)
    
            # dim of pad start from last dim
            input_alt = torch.nn.functional.pad(input,pad=(L,L,L,L) ,mode="constant", value=0)
            output = torch.zeros((input.shape[0],N,C,F,T),dtype=torch.cfloat).to(device)

            for w in range(2*L+1) : 
                for h in range(2*L+1):
                    for n in range(N) : 
                        output[:,n,:,:,:] += torch.mul(
                            input_alt[:,:,w:F-2*L+2+w,h:T-2*L+2+h],
                            filter[:,n,:,w,h,:,:]
                            )
            # iSTFT
            output_raw = torch.zeros((input.shape[0],N,C,batch_data['target'].shape[-1])).to(device)

            # torch does not supprot batch STFT/iSTFT
            for j in range(output_raw.shape[1]) :
                for k in range(output_raw.shape[2]) : 
                # reducing target length due to STFT 1 frame mismatch
                    output_raw[:,j,k,:-shift] = torch.istft(output[:,j,k,:,:],n_fft = hp.model.n_fft)

            ## Normalization
            denom_max = torch.max(torch.abs(output_raw),dim=3)[0]
            denom_max = torch.unsqueeze(denom_max,dim=-1)
            output_raw = output_raw/denom_max


            target = batch_data['target'].to(device)

            ## Metric
            B_SIR = 0
            n_src = batch_data["n_src"][0]

            for C_SIR in range(output_raw.shape[2]):
                SIR_eval[n_src-1] += SIR(output_raw[B_SIR,:,C_SIR,:],target[B_SIR,:,C_SIR,:],device=device)

                cnt_SIR[n_src-1] +=1

                for N_SIR in range(n_src): 
                    PESQ_eval[n_src-1] += PESQ(output_raw[B_SIR,N_SIR,C_SIR,:],target[B_SIR,N_SIR,C_SIR,:])

                    cnt_PESQ[n_src-1] +=1


    print("version : {}".format(args.config))
 
    print("SIR {} | PESQ {}".format(
        torch.sum(SIR_eval)/np.sum(cnt_SIR),
        torch.sum(PESQ_eval)/np.sum(cnt_PESQ)
    ))

    for i in range(4) : 
        SIR_eval[i] /=cnt_SIR[i]
        PESQ_eval[i] /=cnt_PESQ[i]
        print("n_src :  {} | SIR : {} | PESQ : {}".format(i+1,SIR_eval[i],PESQ_eval[i]))
       



        

