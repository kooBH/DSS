import torch
import argparse
import torchaudio
import os
import numpy as np

from tensorboardX import SummaryWriter

from Datasets.DatasetDOA import DatasetDOA
from Datasets.DatasetMIDR import DatasetMIDR
from Datasets.DatasetUDSS import DatasetUDSS
from Datasets.DatasetLRS import DatasetLRS

from ptUtils.hparams import HParam
from ptUtils.writer import MyWriter
from ptUtils.Loss import wSDRLoss,TrunetLoss
from ptUtils.metric import run_metric

from common import run,get_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', type=str, default=None,
                        help="default configuration")
    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--step','-s',type=int,required=False,default=0)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    args = parser.parse_args()

    hp = HParam(args.config,args.default,merge_except=["architecture"])
    print("INFO::Loading configuration : "+args.config)

    device = args.device
    version = args.version_name
    torch.cuda.set_device(device)

    batch_size = hp.train.batch_size
    num_epochs = hp.train.epoch
    num_workers = hp.train.num_workers

    print("INFO::Batch size : {}".format(batch_size))

    best_loss = 1e7

    ## load
    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + version
    log_dir = hp.log.root+'/'+'log'+'/'+version

    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)

    writer = MyWriter(log_dir)

    if hp.dataset== "DOA" : 
        train_dataset = DatasetDOA(hp,is_train = True)
        test_dataset= DatasetDOA(hp,is_train = False)
    elif hp.dataset == "MIDR" :
        train_dataset = DatasetMIDR(hp,is_train = True)
        test_dataset= DatasetMIDR(hp,is_train = False)
    elif hp.dataset == "UDSS" :
        train_dataset = DatasetUDSS(hp,is_train = True)
        test_dataset= DatasetUDSS(hp,is_train = False)
    elif hp.dataset == "LRS" :
        train_dataset = DatasetLRS(hp,is_train = True)
        test_dataset= DatasetLRS(hp,is_train = False)
    else :
        raise Exception("ERROR::Unknown dataset : {}".format(hp.dataset))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)

    print("INFO::Dataset Loaded")

    model = get_model(hp).to(device)

    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        model.load_state_dict(torch.load(args.chkpt, map_location=device))
    if hp.loss.type == "MSELoss":
        criterion = torch.nn.MSELoss()
    elif hp.loss.type == "wSDRLoss" : 
        criterion = wSDRLoss
    elif hp.loss.type == "TrunetLoss" :
        criterion = TrunetLoss(alpha=hp.loss.TrunetLoss.alpha)
    else :
        raise Exception("ERROR::Unsupported criterion : {}".format(hp.loss.type))

    print("INFO::Model Loaded")

    if hp.train.optimizer == 'Adam' :
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.train.Adam)
    elif hp.train.optimizer == 'AdamW' :
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp.train.AdamW.lr)
    else :
        raise Exception("ERROR::Unknown optimizer : {}".format(hp.train.optimizer))

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
    elif hp.scheduler.type == "CosineAnnealingLR" : 
       scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.scheduler.CosineAnnealingLR.T_max, eta_min=hp.scheduler.CosineAnnealingLR.eta_min)
    else :
        raise Exception("ERROR::Unsupported sceduler type : {}".format(hp.scheduler.type))

    step = args.step

    print("INFO::Train starts")
    for epoch in range(num_epochs):
        ### TRAIN ####
        model.train()
        train_loss=0
        for i, (batch_data) in enumerate(train_loader):
            step +=1

            loss = run(hp,device,batch_data,model,criterion)
            optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            train_loss += loss.item()

            if step %  hp.train.summary_interval == 0:
                print('TRAIN::{} : Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(version,epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
                writer.log_value(loss,step,'train loss : '+hp.loss.type)
            
        train_loss = train_loss/len(train_loader)
        torch.save(model.state_dict(), str(modelsave_path)+'/lastmodel.pt')
            
        #### EVAL ####
        model.eval()
        with torch.no_grad():
            test_loss =0.
            for j, (batch_data) in enumerate(test_loader):
                estim, loss = run(hp,device,batch_data,model,criterion,ret_output=True)
                test_loss += loss.item()

            test_loss = test_loss/len(test_loader)

            print('TEST::{} :  Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(version, epoch+1, num_epochs, j+1, len(test_loader), test_loss))

            scheduler.step(test_loss)
            
            writer.log_value(test_loss,step,'test loss : ' + hp.loss.type)

            log_estim = estim[0,:]
            log_clean = batch_data["clean"][0,:]
            log_noisy = batch_data["noisy"][0,0,:]

            writer.log_audio(log_noisy,"noisy_audio",step)
            writer.log_audio(log_estim,"estim_audio",step)
            writer.log_audio(log_clean,"clean_audio",step)

            writer.log_spec(log_noisy,"noisy_spec",step)
            writer.log_spec(log_estim,"estim_spec",step)
            writer.log_spec(log_clean,"clean_spec",step)

            if best_loss > test_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = test_loss

            pesq = 0
            stoi = 0
            for i in range(hp.log.n_eval) : 
                data = test_dataset[i]
                noisy = torch.unsqueeze(data["noisy"].to(device),0)
                clean = torch.unsqueeze(data["clean"].to(device),0)
                angle = torch.unsqueeze(data["angle"].to(device),0)
                mic_pos = torch.unsqueeze(data["mic_pos"].to(device),0)
                face = torch.unsqueeze(data["face"].to(device),0)

                estim  = model(noisy,angle,mic_pos,face).cpu().detach().numpy()
                pesq += run_metric(estim[0],clean[0],"PESQ") 
                stoi += run_metric(estim[0],clean[0],"STOI") 

            pesq /= hp.log.n_eval
            stoi /= hp.log.n_eval

            writer.log_value(pesq,step,'PESQ')
            writer.log_value(stoi,step,'STOI')



