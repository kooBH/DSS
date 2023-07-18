import torch
import torch.nn as nn
#from DSS import DSS
from Attractor.Attractor import DirectionAttractorNet
from UNet.UDSS import UDSS_helper

def get_n_channel(hp):
    ch_in = 0
    n_channel = hp.audio.n_channel

    list_input = hp.model.input
    
    print(" == ch_in == ")
    if "spec" in list_input : 
        ch_in +=2*n_channel
        print("spec : {}".format(ch_in))

    if "mag" in list_input : 
        ch_in +=1
        print("mag : {}".format(ch_in))

    if "LogPowerSpectral" in list_input : 
        ch_in +=1
        print("LogPowerSpectral : {}".format(ch_in))

    if "cossinIPD" in list_input:
        ch_in += 3*(n_channel-1)
        print("cossinIPD : {}".format(ch_in))

    if "cosIPD" in list_input : 
        ch_in += (n_channel-1)
        print("cosIPD : {}".format(ch_in))

    if "AF" in list_input : 
        ch_in += 2
        print("AF : {}".format(ch_in))
    return ch_in

def get_model(hp):
    if hp.model.type == "Attractor" :
        model = DirectionAttractorNet(
            n_channel = hp.audio.n_channel,
            method_out=hp.model.method_out,
            type_activation=hp.model.type_activation,
            type_normalization=hp.model.type_normalization,
            type_activation_out=hp.model.type_activation_out,
            dropout = hp.model.dropout,
            spectral_feature = hp.model.spectral_feature,
            angle_feature = hp.model.angle_feature
            )
    elif hp.model.type == "UDSS" : 
        model = UDSS_helper(n_fft = hp.audio.n_fft,
                            dropout=hp.model.dropout,
                            bottleneck=hp.model.bottleneck,
                            model_complexity = hp.model.complexity,
                            type_encoder = hp.model.type_encoder,
                            use_SV = hp.model.use_SV,
                            corr = hp.model.corr,
                            DSB = hp.model.DSB
                            )
    elif hp.model.type == "UDSSv2" :
        from UDSSv2 import UDSSv2_helper
        model = UDSSv2_helper(encoder = hp.model.encoder)
    elif hp.model.type == "AMTFAA":
        from MTFAA.AMTFAA import AMTFAA_helper
        model = AMTFAA_helper(n_fft = hp.audio.n_fft,
                              corr = hp.model.corr
                              )
    elif hp.model.type == "SMTFAA":
        from MTFAA.SMTFAA import SMTFAA_helper
        model = SMTFAA_helper(n_fft = hp.audio.n_fft,
                              corr = hp.model.corr
                              )
    else :
        pass
        #model = DSS(hp.model.type,n_channel=get_n_channel(hp))
    return model

def run(hp,device,data,model,criterion,ret_output=False): 
    
    if hp.model.type == "Attractor" :
        noisy = data["noisy"].to(device)
        target = data["clean"].to(device)
        estim = model(noisy,data["angle"].to(device),data["mic_pos"].to(device))
    else :
        noisy = data["noisy"].to(device)
        target = data["clean"].to(device)
        estim = model(
            noisy,
            data["angle"].to(device),
            data["mic_pos"].to(device)
        )

    if criterion is None : 
        return estim
    
    if hp.loss.type == "wSDRLoss" : 
        loss = criterion(estim,noisy[:,0,:],target, alpha=hp.loss.wSDRLoss.alpha)
    elif hp.loss.type == "TrunetLoss" :
        loss = criterion(estim,target).to(device)
    else : 
        loss = criterion(estim,target).to(device)

    if ret_output :
        return estim, loss
    else : 
        return loss