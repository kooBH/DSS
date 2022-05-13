import torch
import librosa
import soundfile as sf
import argparse
import os
import numpy as np

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
    parser.add_argument('--dir_output','-o',type=str,required=True)
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

    os.makedirs(args.dir_output,exist_ok=True)

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

            ## Save
            # parse id
            path = batch_data['path_raw'][0]
            name = path.split('/')[-1]
            id = name.split('.')[0]

            for j in range(N) : 
                sf.write(args.dir_output+'/'+id+'_'+str(j)+'.wav',output_raw[0,j,0,:].cpu().detach().numpy(),16000)

        


            

