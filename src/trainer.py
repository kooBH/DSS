import torch
import librosa
import argparse
import torchaudio
import os
import numpy as np

from dataset import DatasetDOA
from cRFConvTasNet import cRFConvTasNet

from ptUtils.hparams import HParam
from ptUtils.writer import MyWriter
from ptUtils.Loss import mSDRLoss,wSDRLoss,SISDRLoss,iSDRLoss



def get_loss(hp,target,output,criterion,device,raw=None): 
    N = hp.model.n_target
    C = hp.data.n_channel

    loss = torch.tensor(0.0).to(device)

    if not hp.loss.cross : 
        for l in range(N) : 
            for c in range(C) : 
                if hp.loss.type == "wSDRLoss":
                    loss +=  Loss(output_raw[:,l,c,:],raw[:,c,:],target[:,l,c,:],alpha = hp.loss.wSDRLoss.alpha).to(device)
                else : 
                    loss += Loss(output_raw[:,l,c,:],target[:,l,c,:]).to(device)
    else : 
        for l in range(N) : 
            for l2 in range(N) : 
                for c in range(C) : 
                    if hp.loss.type == "wSDRLoss":
                        val_loss =  Loss(output_raw[:,l,c,:],raw[:,c,:],target[:,l2,c,:],alpha = hp.loss.wSDRLoss.alpha).to(device)
                    else : 
                        val_loss = Loss(output_raw[:,l,c,:],target[:,l2,c,:]).to(device)
                    
                    if l == l2 : 
                        loss += val_loss
                    else :
                        loss -= val_loss

    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--step','-s',type=int,required=False,default=0)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    args = parser.parse_args()

    hp = HParam(args.config)
    print("NOTE::Loading configuration : "+args.config)

    device = args.device
    version = args.version_name
    torch.cuda.set_device(device)

    batch_size = hp.train.batch_size
    num_epochs = hp.train.epoch
    num_workers = hp.train.num_workers

    best_loss = 1e7

    n_target = hp.model.n_target

    ## load
    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + version
    log_dir = hp.log.root+'/'+'log'+'/'+version

    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)

    writer = MyWriter(hp, log_dir)

    ## Dataloader
    #dataset_train = DatasetDOA(hp.data.root+"/train")
    #dataset_test  = DatasetDOA(hp.data.root+"/test")
    dataset_train = DatasetDOA(hp.data.root_train,n_target=n_target)
    dataset_test  = DatasetDOA(hp.data.root_test,n_target=n_target)

    print("len_trainset : {}".format(len(dataset_train)))

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    model = cRFConvTasNet(
        L=hp.model.l_filter,
        f_ch=hp.model.d_feature,
        n_fft=hp.model.n_fft,
        mask=hp.model.activation,
        n_target=n_target
    ).to(device)

    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        model.load_state_dict(torch.load(args.chkpt, map_location=device))

    ## Loss ##
    if hp.loss.type == "wSDRLoss":
        Loss = wSDRLoss
    elif hp.loss.type == "mSDRLoss":
        Loss = mSDRLoss
    elif hp.loss.type =="SISDRLoss":
        Loss = SISDRLoss
    elif hp.loss.type =="iSDRLoss":
        Loss = iSDRLoss
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

    N = n_target
    L = hp.model.l_filter
    C = hp.data.n_channel
    F = int(hp.model.n_fft/2 + 1)
    T = hp.data.n_frame

    shift = int(hp.model.n_fft/4)

    step = args.step

    idx_plot = np.random.randint(hp.train.batch_size)

    for epoch in range(num_epochs) :
        ### TRAIN ####
        model.train()
        train_loss=0
        for i, (batch_data) in enumerate(train_loader):
            step +=1
            
            ## run model
            feature = batch_data['flat'].to(device)
            # output = [B,C, 2*L+1,2*L+1, n_hfft, Time]
            filter = model(feature)

            ## filtering
            # [B,C,F,T]
            input = batch_data['spec'].to(device)
            """
            torch.nn.functional.pad(input, pad, mode='constant', value=0.0)
            https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html

            " When using the CUDA backend, this operation may induce nondeterministic behaviour in its backward pass that is not easily switched off. Please see the notes on Reproducibility for background."

            Would it be a trouble? 
            """
            # dim of pad start from last dim
            input_alt = torch.nn.functional.pad(input,pad=(L,L,L,L) ,mode="constant", value=0)
            output = torch.zeros((input.shape[0],N,C,F,T),dtype=torch.cfloat).to(device)

            ## TODO : there should be fancier way to do this.
            for w in range(2*L+1) : 
                for h in range(2*L+1):
                    for n in range(N) : 
                        output[:,n,:,:,:] += torch.mul(
                            input_alt[:,:,w:F-2*L+2+w,h:T-2*L+2+h],
                            filter[:,n,:,w,h,:,:]
                            )
            
            # iSTFT
            output_raw = torch.zeros((input.shape[0],N,C,batch_data['target'].shape[-1])).to(device)

            # torch STFT/iSTFT
            for j in range(output_raw.shape[1]) :
                for k in range(output_raw.shape[2]) : 
                # reducing target length due to STFT 1 frame mismatch
                    output_raw[:,j,k,:-shift] = torch.istft(output[:,j,k,:,:],n_fft = hp.model.n_fft)

            ## Normalization
            denom_max = torch.max(torch.abs(output_raw),dim=3)[0]
            denom_max = torch.unsqueeze(denom_max,dim=-1)
            output_raw = output_raw/denom_max

            ## Loss
            target = batch_data['target'].to(device)
            raw = batch_data['raw'].to(device)

            loss = get_loss(hp,target,output_raw,Loss,device,raw)

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
            test_loss =0.
            for i, (batch_data) in enumerate(test_loader):
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

                ## Loss
                target = batch_data['target'].to(device)
                raw = batch_data['raw'].to(device)
                loss = get_loss(hp,target,output_raw,Loss,device,raw)

                print('TEST::{} : Epoch [{}/{}], Step [{}/{}], Loss: {:.4e}'.format(version, epoch+1, num_epochs, i+1, len(test_loader), loss.item()))
                test_loss +=loss.item()




            test_loss = test_loss/len(test_loader)
            if hp.scheduler.type == 'Plateau':
                scheduler.step(test_loss)
            else :
                scheduler.step(test_loss)
            
            writer.log_value(test_loss,step,'test loss : ' + hp.loss.type)

            writer.log_audio(raw[idx_plot,0,:],"input {}".format(batch_data['path_raw'][idx_plot]),step)

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

                
                
                
                

