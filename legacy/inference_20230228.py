import torch
import librosa
import soundfile as sf
import argparse
import os
import numpy as np

from tqdm import tqdm

from dataset import DatasetDOA
from cRFConvTasNet import cRFConvTasNet
from cRFUNet import UNet10, UNet20
from dataset import preprocess,get_n_feature,deemphasis
from ptUtils.hparams import HParam


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--chkpt',type=str,required=True)
    parser.add_argument('--dir_input','-i',type=str,required=True)
    parser.add_argument('--dir_output','-o',type=str,required=True)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    args = parser.parse_args()


    hp = HParam(args.config)
    preemphasis=hp.data.preemphasis
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
    T = hp.data.n_frame
    shift = int(hp.model.n_fft/4)

    modelsave_path = args.chkpt

    os.makedirs(args.dir_output,exist_ok=True)

    n_feature = get_n_feature(
        hp.data.n_channel,
        hp.model.n_target, 
        mono=hp.model.mono, 
        phase = hp.model.phase,
        full_phase = hp.model.phase_full
        )

    ##  Model
    if hp.model.type == "ConvTasNet":
        model = cRFConvTasNet(
            n_feature=n_feature,
            L_t=hp.model.l_filter_t,
            L_f=hp.model.l_filter_f,
            f_ch=hp.model.d_feature,
            n_fft=hp.model.n_fft,
            mask=hp.model.activation,
            n_target=n_target,
            hp=hp
        ).to(device)
        flat = True
    elif hp.model.type == "UNet10" :
        model = UNet10(
            c_in=n_feature,
            c_out=n_target,
            L_t=hp.model.l_filter_t,
            L_f=hp.model.l_filter_f,
            n_fft=hp.model.n_fft,
            device=device,
            mask=hp.model.activation
        ).to(device)
        flat = False
    elif hp.model.type == "UNet20" :
        model = UNet20(
            c_in=n_feature,
            c_out=n_target,
            L_t=hp.model.l_filter_t,
            L_f=hp.model.l_filter_f,
            n_fft=hp.model.n_fft,
            device=device,
            mask=hp.model.activation
        ).to(device)
        flat = False
    else :
        raise Exception("ERROR:: Unknown Model {}".format(hp.model.type))

    dataset = DatasetDOA(
        args.dir_input,
        n_target,
        IPD=hp.model.phase,
        mono = hp.model.mono,
        LPS = hp.model.LPS,
        full_phase=hp.model.phase_full,
        #preemphasis=hp.data.preemphasis ,
        #preemphasis_coef =hp.data.preemphasis_coef ,
        #preemphasis_order =hp.data.preemphasis_order,
        flat=flat,
        only_azim=hp.model.only_azim,
        ADPIT=hp.model.ADPIT
        )
    loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,num_workers=num_workers)

    model.load_state_dict(torch.load(args.chkpt, map_location=device))

    #### EVAL ####
    model.eval()
    with torch.no_grad():
        test_loss =0.
        for i, (batch_data) in tqdm(enumerate(loader),total=len(dataset)):
            # run model
            feature = batch_data['flat'].float().to(device)
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

            ## Save
            # parse id
            path = batch_data['path_raw'][0]
            name = path.split('/')[-1]
            id = name.split('.')[0]

            output_raw = output_raw.cpu().detach().numpy()

            for j in range(N) : 
                if preemphasis : 
                    t_out = dataset.deemphasis((output_raw[0,j,:,:]))
                    sf.write(args.dir_output+'/'+id+'_'+str(j)+'.wav',t_out.T,16000)
                else :
                    sf.write(args.dir_output+'/'+id+'_'+str(j)+'.wav',output_raw[0,j,:,:].T,16000)

        


            

