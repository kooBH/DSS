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
from ptUtils.Loss import SISDR


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
    torch.cuda.set_device(device)

    batch_size = hp.train.batch_size
    num_epochs = hp.train.epoch
    num_workers = hp.train.num_workers

    best_loss = 10

    ## load
    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + args.version_name
    log_dir = hp.log.root+'/'+'log'+'/'+args.version_name

    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)

    writer = MyWriter(hp, log_dir)

    ## Dataloader
    #dataset_train = DatasetDOA(hp.data.root+"/train")
    #dataset_test  = DatasetDOA(hp.data.root+"/test")
    dataset_train = DatasetDOA(hp.data.root)
    dataset_test  = DatasetDOA(hp.data.root)

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    # TODO
    model = cRFConvTasNet(
        L=hp.model.l_filter,
        f_ch=hp.model.d_feature,
        n_fft=hp.model.n_fft,
        mask=hp.model.activation
    ).to(device)

    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        model.load_state_dict(torch.load(args.chkpt, map_location=device))

    # TODO
    ## Loss ##
    Loss = SISDR

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

    B = hp.train.batch_size
    N = hp.model.n_target
    L = hp.model.l_filter
    C = hp.data.n_channel
    F = int(hp.model.n_fft/2 + 1)
    T = hp.data.n_frame

    shift = int(hp.model.n_fft/4)

    step = args.step

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
            output = torch.zeros((B,N,C,F,T),dtype=torch.cfloat).to(device)


            ## TODO : there should be fancier way to do this.
            for w in range(2*L+1) : 
                for h in range(2*L+1):
                    for n in range(N) : 
                        output[:,n,:,:,:] += torch.mul(
                            input_alt[:,:,w:F-2*L+2+w,h:T-2*L+2+h],
                            filter[:,n,:,w,h,:,:]
                            )
            # iSTFT
            output_raw = torch.zeros((B,N,C,batch_data['target'].shape[-1])).to(device)

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
            loss = Loss(output_raw,target).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss+=loss.item()
            print('TRAIN::Epoch [{}/{}], Step [{}/{}], Loss: {:.4e}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

            if step %  hp.train.summary_interval == 0:
                writer.log_value(loss,step,'train loss : '+hp.loss.type)

        train_loss = train_loss/len(train_loader)
        torch.save(model.state_dict(), str(modelsave_path)+'/lastmodel.pt')
            
        #### EVAL ####
        model.eval()
        with torch.no_grad():
            test_loss =0.
            for j, (batch_data) in enumerate(test_loader):
                # run model
                feature = batch_data['flat'].to(device)
                # output = [B,C, 2*L+1,2*L+1, n_hfft, Time]
                filter = model(feature)

                ## filtering
                # [B,C,F,T]
                input = batch_data['spec'].to(device)
       
                # dim of pad start from last dim
                input_alt = torch.nn.functional.pad(input,pad=(L,L,L,L) ,mode="constant", value=0)
                output = torch.zeros((B,N,C,F,T),dtype=torch.cfloat).to(device)

                for w in range(2*L+1) : 
                    for h in range(2*L+1):
                        for n in range(N) : 
                            output[:,n,:,:,:] += torch.mul(
                                input_alt[:,:,w:F-2*L+2+w,h:T-2*L+2+h],
                                filter[:,n,:,w,h,:,:]
                                )
                # iSTFT
                output_raw = torch.zeros((B,N,C,batch_data['target'].shape[-1])).to(device)

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

                loss = Loss(output_raw,target).to(device)

                print('TEST::Epoch [{}/{}], Step [{}/{}], Loss: {:.4e}'.format(epoch+1, num_epochs, j+1, len(test_loader), loss.item()))
                test_loss +=loss.item()

            test_loss = test_loss/len(test_loader)
            if hp.scheduler.type == 'Plateau':
                scheduler.step(test_loss)
            else :
                scheduler.step(test_loss)
            
            writer.log_value(test_loss,step,'test loss : ' + hp.loss.type)

      

            if best_loss > test_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = test_loss

        idx = 3
        raw,_ = librosa.load(batch_data["path_raw"][idx],sr=16000,mono=False)
        raw = raw[0,:]

        writer.log_audio(raw,"input",step)
        for i in range(4) : 
            writer.log_audio(output_raw[idx,i,0,:].cpu().detach().numpy(),"output {}".format(i),step)
            writer.log_audio(target[idx,i,0,:].cpu().detach().numpy(),"target {}".format(i),step)
        