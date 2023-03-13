import torch
import librosa
import soundfile as sf
import argparse
import os
import numpy as np
from ptUtils.metric import SIR,PESQ

from tqdm import tqdm

from dataset import DatasetDOA,get_n_feature,deemphasis
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

    np.random.seed(0)

    hp = HParam(args.config)
    print("NOTE::Loading configuration : "+args.config)
    n_target = hp.model.n_target

    device = args.device
    torch.cuda.set_device(device)

    batch_size = 1
    num_workers = 1

    N = n_target
    L_t = hp.model.l_filter_t
    L_f = hp.model.l_filter_f
    C = hp.data.n_channel
    F = int(hp.model.n_fft/2 + 1)
    ## TODO : have to be availiable to arbitary length of audio
    #T = hp.data.n_frame
    T = 125
    #T = 500
    shift = int(hp.model.n_fft/4)

    modelsave_path = args.chkpt

    dataset = DatasetDOA(args.dir_input,n_target,
        IPD=hp.model.phase,
        mono=hp.model.mono,
        LPS=hp.model.LPS,
        full_phase=hp.model.phase_full
    )
    loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,num_workers=num_workers)

    n_feature = get_n_feature(hp.data.n_channel,hp.model.n_target, hp.model.mono, hp.model.phase,hp.model.phase_full)

    model = cRFConvTasNet(
        n_feature=n_feature,
        L_t=hp.model.l_filter_t,
        L_f=hp.model.l_filter_f,
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

            # Deemphasis
            if hp.data.preemphasis :
                for it_B in range(output_raw.shape[0]) :
                    for it_N in range(N) : 
                        output_raw[it_B,it_N] = (deemphasis(output_raw[it_B,it_N].T,device=device)).T

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
    if cnt_PESQ[0] == 0 : 
        print("SIR {} | PESQ {}".format(
            torch.sum(SIR_eval[1:])/np.sum(cnt_SIR[1:]),
            torch.sum(PESQ_eval)/np.sum(cnt_PESQ[1:])
        ))
    else :
        print("SIR {} | PESQ {}".format(
            torch.sum(SIR_eval[1:])/np.sum(cnt_SIR[1:]),
            torch.sum(PESQ_eval)/np.sum(cnt_PESQ[:])
        ))

    for i in range(4) : 
        SIR_eval[i] /=cnt_SIR[i]
        PESQ_eval[i] /=cnt_PESQ[i]
        if i != 0 : 
            print("n_src :  {} | SIR : {} | PESQ : {}".format(i+1,SIR_eval[i],PESQ_eval[i]))
        else :
            print("n_src :  {} | SIR : -- | PESQ : {}".format(i+1,PESQ_eval[i]))

       



        

