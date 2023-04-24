"""
Inferece for test dataset
"""

import torch
import argparse
import os
import numpy as np

from Datasets.DatasetDOA import DatasetDOA
from Datasets.DatasetMIDR import DatasetMIDR

from ptUtils.hparams import HParam
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
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    args = parser.parse_args()

    hp = HParam(args.config,args.default)
    print("NOTE::Loading configuration : "+args.config)

    device = args.device
    version = args.version_name
    torch.cuda.set_device(device)

    batch_size = hp.train.batch_size
    num_epochs = hp.train.epoch
    num_workers = hp.train.num_workers

    ## load
    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + version

    if hp.dataset== "DOA" : 
        test_dataset= DatasetDOA(hp,is_train = False)
    elif hp.dataset == "MIDR" :
        test_dataset= DatasetMIDR(hp,is_train = False)
    else :
        raise Exception("ERROR::Unknown dataset : {}".format(hp.dataset))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)

    model = get_model(hp).to(device)

    print('NOTE::Loading pre-trained model : '+ args.chkpt)
    model.load_state_dict(torch.load(args.chkpt, map_location=device))


    #### EVAL ####
    model.eval()
    with torch.no_grad():
        test_loss =0.
        for j, (batch_data) in enumerate(test_loader):
            estim= run(hp,device,batch_data,model,None,
                        ret_output=True)
        """
        pesq = 0
        for i in range(hp.log.n_eval) : 
            data = test_dataset[i]
            noisy = torch.unsqueeze(data["noisy"].to(device),0)
            clean = torch.unsqueeze(data["clean"].to(device),0)
            angle = torch.unsqueeze(data["angle"].to(device),0)
            mic_pos = torch.unsqueeze(data["mic_pos"].to(device),0)

            estim  = model(noisy,angle,mic_pos).cpu().detach().numpy()
            pesq += run_metric(estim[0],clean[0],"PESQ") 
        pesq /= hp.log.n_eval
        """




