import torch
import librosa
import argparse
import torchaudio
import os
import numpy as np

from dataset import DatasetDOA,get_n_feature
from cRFConvTasNet import cRFConvTasNet
from cRFUNet import UNet10, UNet20
from UNet.UNet import UNet

from ptUtils.hparams import HParam
from ptUtils.writer import MyWriter
from ptUtils.Loss import mSDRLoss,wSDRLoss,SISDRLoss,iSDRLoss,wMSELoss
from ptUtils.metric import SIR,PESQ

def get_wav_loss(hp,target,output_raw,n_src,criterion,device,raw=None): 
    C = hp.data.n_channel

    loss = torch.tensor(0.0,requires_grad=True).to(device)
    
    if hp.model.ADPIT : 
        for l in range(N) : 
            for c in range(c_out) : 
                if hp.loss.type == "wSDRLoss":
                    loss +=  Loss(output_raw[:,l,c,:],raw[:,c,:],target[:,l,c,:],alpha = hp.loss.wSDRLoss.alpha).to(device)
                else : 
                    loss += Loss(output_raw[:,l,c,:],target[:,l,c,:]).to(device)
    else : 
        for b in range(target.shape[0]) : 
            N_temp = n_src[b]
            for c in range(c_out) : 
                if hp.loss.type == "wSDRLoss":
                    loss +=  Loss(output_raw[b,:N_temp,c,:],raw[b,c,:],target[b,:N_temp,c,:],alpha = hp.loss.wSDRLoss.alpha).to(device)
                else : 
                    loss += Loss(output_raw[b,:N_temp,c,:],target[b,:N_temp,c,:]).to(device)

    return loss

def get_spectral_loss(hp,target_spec,output_spec,n_src,criterion,device): 
    C = hp.data.n_channel

    loss = torch.tensor(0.0,requires_grad=True).to(device)

    if hp.model.ADPIT : 
        for l in range(N) : 
            for c in range(C) : 
                if hp.loss.type == "wMSELoss" : 
                    loss += Loss(output_spec[:,l,c,:],target_spec[:,l,c,:],alpha=hp.loss.wMSELoss.alpha).to(device)
                else :
                    loss += Loss(output_spec[:,l,c,:],target_spec[:,l,c,:]).to(device)
    else : 
        for b in range(target.shape[0]) : 
            N_temp = n_src[b]
            for c in range(C) : 
                if hp.loss.type == "wMSELoss" : 
                    loss += Loss(output_spec[b,:N_temp,c,:],target_spec[b,:N_temp,c,:],alpha=hp.loss.wMSELoss.alpha).to(device)
                else :
                    loss += Loss(output_spec[b,:N_temp,c,:],target_spec[b,:N_temp,c,:]).to(device)
    return loss

def run(model,feature,input,target,raw,n_src, ret_output=False) :

    # output = [B,C, 2*L_f+1,2*L_t+1, n_hfft, Time]
    filter = model(feature)

    ## filtering
    """
    torch.nn.functional.pad(input, pad, mode='constant', value=0.0)
    https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html

    " When using the CUDA backend, this operation may induce nondeterministic behaviour in its backward pass that is not easily switched off. Please see the notes on Reproducibility for background."

    Would it be a trouble? 
    """
    # dim of pad start from last dim
    input_alt = torch.nn.functional.pad(input,pad=(L_t,L_t,L_f,L_f) ,mode="constant", value=0)
    output = torch.zeros((input.shape[0],N,c_out,F,T),dtype=torch.cfloat).to(device)

    ## TODO : there should be fancier way to do this.
    for t in range(2*L_t+1) : 
        for f in range(2*L_f+1):
            for n in range(N) : 
                output[:,n,:c_out,:,:] += torch.mul(
                    input_alt[: , :c_out, f:F+f , t:T+t ],
                    filter[:,n,:c_out,f,t,:,:]
                    )
    
    # iSTFT
    output_raw = torch.zeros((input.shape[0],N,c_out,target.shape[-1])).to(device)

    # torch STFT/iSTFT
    for j in range(N) :
        for k in range(c_out) : 
        # reducing target length due to STFT 1 frame mismatch
            output_raw[:,j,k,:-shift] = torch.istft(output[:,j,k,:,:],n_fft = hp.model.n_fft)

    ## Normalization
    denom_max = torch.max(torch.abs(output_raw),dim=3)[0]
    denom_max = torch.unsqueeze(denom_max,dim=-1)
    output_raw = output_raw/(denom_max + 1e-7)

    ## Loss
    if hp.loss.type == "wSDRLoss" : 
        loss = get_wav_loss(hp,target,output_raw,n_src,Loss,device,raw)
    elif hp.loss.type == "wMSELoss" : 
        target_spec = torch.zeros_like(output).to(device)
        for i in range(output.shape[0]) : 
            for j in range(N) : 
                target_spec[i,j,:,:,:] = torch.stft(target[i,j,:,:],n_fft=hp.model.n_fft,return_complex=True)[:,:,:-1]

        loss = get_spectral_loss(hp,target_spec,output,n_src,Loss,device)
    else : 
        raise Exception("ERROR::run():The loss type is not supported | {}".format(hp.loss.type))

    if not ret_output : 
        return loss
    else : 
        return loss, output_raw

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,help="yaml for configuration")
    parser.add_argument('--default', type=str, required=False,default=None,help="base yaml")
    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--step','-s',type=int,required=False,default=0)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    args = parser.parse_args()

    global hp
    hp = HParam(args.config,args.default)
    print("NOTE::Loading configuration : "+args.config)

    global N, L_t, L_f, C,F,T,shift,n_fft,c_out
    global device, n_target

    n_target = hp.model.n_target
    device = args.device
    version = args.version_name
    torch.cuda.set_device(device)
    batch_size = hp.train.batch_size
    num_epochs = hp.train.epoch
    num_workers = hp.train.num_workers

    best_loss = 1e7

    N = n_target
    L_t = hp.model.l_filter_t
    L_f = hp.model.l_filter_f
    C = hp.data.n_channel
    F = int(hp.model.n_fft/2 + 1)
    T = hp.data.n_frame
    n_fft = hp.model.n_fft
    c_out=1




    ## load
    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + version
    log_dir = hp.log.root+'/'+'log'+'/'+version

    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)

    writer = MyWriter(hp, log_dir)

    n_feature = get_n_feature(hp.data.n_channel,hp.model.n_target, 
    hp.model.mono, 
    hp.model.phase,
    hp.model.phase_full
    )

##  Model
    if hp.model.type == "ConvTasNet":
        model = cRFConvTasNet(
            n_feature=n_feature,
            c_out=c_out,
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

    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        model.load_state_dict(torch.load(args.chkpt, map_location=device))

    ## Dataloader
    #dataset_train = DatasetDOA(hp.data.root+"/train")
    #dataset_test  = DatasetDOA(hp.data.root+"/test")
    dataset_train = DatasetDOA(hp.data.root_train,
    n_target=n_target,
    LPS = hp.model.LPS,
    ADPIT=hp.model.ADPIT,
    preemphasis_coef =hp.data.preemphasis_coef ,
    preemphasis_order =hp.data.preemphasis_order,
    azim_shaking=hp.model.azim_shaking
    )
    dataset_test  = DatasetDOA(hp.data.root_test,
    n_target=n_target,
    LPS = hp.model.LPS,
    ADPIT=hp.model.ADPIT,
    preemphasis_coef =hp.data.preemphasis_coef ,
    preemphasis_order =hp.data.preemphasis_order,
    azim_shaking=hp.model.azim_shaking
    )

    print("len_trainset : {}".format(len(dataset_train)))

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True)

    ## Loss ##
    if hp.loss.type == "wSDRLoss":
        Loss = wSDRLoss
    elif hp.loss.type == "mSDRLoss":
        Loss = mSDRLoss
    elif hp.loss.type =="SISDRLoss":
        Loss = SISDRLoss
    elif hp.loss.type =="iSDRLoss":
        Loss = iSDRLoss
    elif hp.loss.type =="wMSELoss":
        Loss = wMSELoss
    else :
        raise Exception("ERROR::unsupported loss : {}".format(hp.loss.type))

    ## Optimizer ## 
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.train.adam)
    
    ## Scheduler  ##
    if hp.scheduler.type == 'Plateau': 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            mode=hp.scheduler.Plateau.mode,
            factor=hp.scheduler.Plateau.factor,
            patience=hp.scheduler.Plateau.patience,
            min_lr=hp.scheduler.Plateau.min_lr)
    elif hp.scheduler.type == 'oneCycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                max_lr = hp.scheduler.oneCycle.max_lr,
                epochs=hp.train.epoch,
                steps_per_epoch = len(train_loader)
        )
    else :
        raise Exception("Unsupported sceduler type")



    shift = int(hp.model.n_fft/4)

    step = args.step

    print("INFO::Learning")
    for epoch in range(num_epochs) :
        ### TRAIN ####
        model.train()
        train_loss=0
        for i, (batch_data) in enumerate(train_loader):
            step += batch_data['feature'].shape[0]
            
            ## run model
            feature = batch_data['feature'].to(device)

            # [B,C,F,T]
            input = batch_data['spec'].to(device)
            target = batch_data['target'].to(device)
            target = target[:,:,:c_out,:]
            raw = batch_data['raw'].to(device)
            n_src = batch_data['n_src'].to(device)

            loss = run(model,feature,input,target,raw,n_src,ret_output=False)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss+=loss.item()
            print('TRAIN::{} : Epoch [{}/{}], Step [{}/{}], Loss: {:.4e}'.format(version,epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

            if step %  hp.train.summary_interval == 0:
                writer.log_value(loss,step,'train loss : '+hp.loss.type)

        train_loss = train_loss/len(train_loader)
        torch.save(model.state_dict(), str(modelsave_path)+'/lastmodel.pt')
            
        #### EVAL ####
        model.eval()
        with torch.no_grad():
            SIR_eval = torch.tensor(0.0).to(device)
            PESQ_eval = torch.tensor(0.0).to(device)
            cnt_SIR = 0
            cnt_PESQ = 0

            test_loss = torch.tensor(0.0,requires_grad=True)
            for i, (batch_data) in enumerate(test_loader):

                # run model
                feature = batch_data['feature'].to(device)
                input = batch_data['spec'].to(device)
                target = batch_data['target'].to(device)
                target = target[:,:,:c_out,:]
                raw = batch_data['raw'].to(device)
                n_src = batch_data['n_src'].to(device)

                loss,output_raw = run(model,feature,input,target,raw,n_src,ret_output=True)

                """ Too Slow
                ## Metric
                for B_SIR in range(output_raw.shape[0]) :
                    for C_SIR in range(output_raw.shape[2]):
                        SIR_eval += SIR(output_raw[B_SIR,:,C_SIR,:],target[B_SIR,:,C_SIR,:],device=device)
                        cnt_SIR +=1

                        for N_SIR in range(output_raw.shape[1]): 
                            PESQ_eval += PESQ(output_raw[B_SIR,N_SIR,C_SIR,:],target[B_SIR,N_SIR,C_SIR,:])
                            cnt_PESQ+=1
                """                

                ## LOG

                print('TEST::{} : Epoch [{}/{}], Step [{}/{}], Loss: {:.4e}'.format(version, epoch+1, num_epochs, i+1, len(test_loader), loss.item()))
                test_loss +=loss.item()

            test_loss = test_loss/len(test_loader)
            if hp.scheduler.type == 'Plateau':
                scheduler.step(test_loss)
            else :
                scheduler.step(test_loss)

            ## Log 
            np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
            idx_plot = np.random.randint(input.shape[0])

            writer.log_value(test_loss,step,'test loss : ' + hp.loss.type)

            #SIR_eval /=cnt_SIR
            #PESQ_eval /=cnt_PESQ

            #writer.log_value(SIR_eval,step,"SIR")
            #writer.log_value(PESQ_eval,step,"PESQ")

            writer.log_audio(raw[idx_plot,0,:],"input",step)

            plot_data = torch.zeros(8,raw.shape[2])

            for j in range(N) : 
                writer.log_audio(output_raw[idx_plot,j,0,:].cpu().detach().numpy(),"output {}".format(j),step)
                writer.log_audio(target[idx_plot,j,0,:].cpu().detach().numpy(),"target {}".format(j),step)

                plot_data[0+j,:]=target[idx_plot,j,0,:]
                plot_data[4+j,:]=output_raw[idx_plot,j,0,:]
            
            plot_data = plot_data.cpu().detach().numpy()

            writer.log_DOA_wav(plot_data,step)

            if best_loss > test_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = test_loss

                
                
                
                

