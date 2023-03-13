import torch
import librosa
import argparse
import os
import numpy as np

from dataset2 import DatasetDOA2

from ptUtils.hparams import HParam
from ptUtils.writer import MyWriter
from ptUtils.Loss import Loss_mag_wav,wSDRLoss

from cRFUNet import UNet20
from UNet.ResUNet import ResUNet

def run(model,feature, criterion, target,ret_output=False) :
    # output = [B, 2, F, T]
    output = model(feature)
    if hp.loss.type == "wSDRLoss" : 
        loss = criterion(output,feature[:,:2,:,:],target)
    else :
        loss = criterion(output,target)

    if ret_output :
        return loss, output
    else :
        return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,help="yaml for configuration")
    parser.add_argument('--default', type=str, required=False,default=None,help="base yaml")
    parser.add_argument('--version_name', '-v', type=str, required=True, help="version of current training")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--step','-s',type=int,required=False,default=0)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    args = parser.parse_args()

    global hp
    hp = HParam(args.config,args.default)
    print("NOTE::Loading configuration : "+args.config)

    global device, n_target
    device = args.device
    version = args.version_name
    torch.cuda.set_device(device)

    batch_size = hp.train.batch_size
    num_epochs = hp.train.epoch
    num_workers = hp.train.num_workers

    best_loss = 1e7

    ## load
    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + version
    log_dir = hp.log.root+'/'+'log'+'/'+version

    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)

    writer = MyWriter(hp, log_dir)

    ##  Model
    if hp.model.type == "ResUNet":
        model = ResUNet(c_in=10,c_out=2).to(device)
    elif hp.model.type == "DenseUNet":
        raise Exception("ERROR::Unimplemented {}".format(hp.model.type))
    elif hp.model.type == "UNet20":
        model = UNet20(c_in=10,c_out=2,n_target=1).to(device)
    else :
        raise Exception("ERROR:: Unknown Model {}".format(hp.model.type))

    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        model.load_state_dict(torch.load(args.chkpt, map_location=device))

    ## Dataloader
    dataset_train = DatasetDOA2(hp.data.root_train,azim_shaking=hp.model.azim_shaking)
    dataset_test = DatasetDOA2(hp.data.root_test,azim_shaking=hp.model.azim_shaking)
    
    print("len_trainset : {}".format(len(dataset_train)))

    train_loader = torch.utils.data.DataLoader(dataset=dataset_train,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test,batch_size=batch_size,shuffle=False,num_workers=num_workers)

    ## Loss ##
    if hp.loss.type == "Loss_mag_wav":
        criterion = Loss_mag_wav
    elif hp.loss.type == "wSDRLoss" : 
        criterion = wSDRLoss
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

    step = args.step
    torch.autograd.set_detect_anomaly(True)
    print("INFO::Learning Starts")
    for epoch in range(num_epochs) :
        ### TRAIN ####
        model.train()
        train_loss=0
        for i, (batch_data) in enumerate(train_loader):
            step += batch_data['input'].shape[0]

            feature = batch_data["input"].to(device)
            target = batch_data["target"].to(device)

            loss = run(model,feature,criterion,target,ret_output=False)

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
            test_loss = torch.tensor(0.0,requires_grad=True)
            for i, (batch_data) in enumerate(test_loader):

                # run model
                feature = batch_data["input"].to(device)
                target = batch_data["target"].to(device)
               
                loss,output = run(model,feature,criterion,target,ret_output=True)

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
            idx_plot = np.random.randint(feature.shape[0])

            writer.log_value(test_loss,step,'test loss : ' + hp.loss.type)

            # [2,F,T]
            def tensor2mag(ten):
                mag = torch.sqrt(torch.pow(ten[0,:,:],2)+torch.pow(ten[1,:,:],2))
                return mag

            writer.log_spec(tensor2mag(feature[idx_plot,:2,:,:]),'noisy',step)
            writer.log_spec(tensor2mag(target[idx_plot,:,:,:]),'clean',step)
            writer.log_spec(tensor2mag(output[idx_plot]),'estim',step)

            if best_loss > test_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = test_loss

                
                
                
                

