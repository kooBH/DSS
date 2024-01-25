import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from lip.densetcn import *
from lip.resnet import *
from lip.preprocess import *

# -- auxiliary functions
def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)

def calculateNorm2(model):
    para_norm = 0.
    for p in model.parameters():
        para_norm += p.data.norm(2)
    print('2-norm of the neural network: {:.4f}'.format(para_norm**.5))

def _average_batch(x, lengths, B):
    return torch.stack( [torch.mean( x[index][:,0:i], 1 ) for index, i in enumerate(lengths)],0 )

class DenseTCN(nn.Module):
    def __init__( self, block_config, growth_rate_set, input_size, reduced_size, num_classes,
                  kernel_size_set, dilation_size_set,
                  dropout, relu_type,
                  squeeze_excitation=False,
        ):
        super(DenseTCN, self).__init__()

        num_features = reduced_size + block_config[-1]*growth_rate_set[-1]
        self.tcn_trunk = DenseTemporalConvNet( block_config, growth_rate_set, input_size, reduced_size,
                                          kernel_size_set, dilation_size_set,
                                          dropout=dropout, relu_type=relu_type,
                                          squeeze_excitation=squeeze_excitation,
                                          )
        self.tcn_output = nn.Linear(num_features, num_classes)
        self.consensus_func = _average_batch

    def forward(self, x, lengths, B):
        x = self.tcn_trunk(x.transpose(1, 2))
        x = self.consensus_func( x, lengths, B )
        return self.tcn_output(x)

class Lipreading(nn.Module):
    # configs/lrw_resnet18_dctcn_boundary.json
    def __init__(self, 
                 modality='video', 
                 hidden_dim=256, 
                 backbone_type='resnet', 
                 num_classes=500,
                 relu_type='swish', 
                 tcn_options={}, 
                 densetcn_options={
                 "block_config": [
                    3,
                    3,
                    3,
                    3
                ],
                "growth_rate_set": [
                    384,
                    384,
                    384,
                    384
                ],
                "kernel_size_set": [
                    3,
                    5,
                    7
                ],
                "dilation_size_set": [
                    1,
                    2,
                    5
                ],
                "reduced_size": 512,
                "squeeze_excitation": True,
                "dropout": 0.2,
                 }, 
                 width_mult=1.0,
                 use_boundary=False, 
                 extract_feats=True
                 ):
        super(Lipreading, self).__init__()
        self.extract_feats = extract_feats
        self.backbone_type = backbone_type
        self.modality = modality
        self.use_boundary = use_boundary


        self.frontend_nout = 64
        self.backend_out = 512
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)

        # -- frontend3D
        if relu_type == 'relu':
            frontend_relu = nn.ReLU(True)
        elif relu_type == 'prelu':
            frontend_relu = nn.PReLU( self.frontend_nout )
        elif relu_type == 'swish':
            frontend_relu = Swish()

        self.frontend3D = nn.Sequential(
                    nn.Conv3d(1, self.frontend_nout, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
                    nn.BatchNorm3d(self.frontend_nout),
                    frontend_relu,
                    nn.MaxPool3d( kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        
        # -- initialize
        self._initialize_weights_randomly()

    def forward(self, x, boundaries=None):
        B, C, T, H, W = x.size()
        x = self.frontend3D(x)
        Tnew = x.shape[2]    # outpu should be B x C2 x Tnew x H x W
        x = threeD_to_2D_tensor( x )
        x = self.trunk(x)

        if self.backbone_type == 'shufflenet':
            x = x.view(-1, self.stage_out_channels)
        x = x.view(B, Tnew, x.size(1))

        # -- duration
        if self.use_boundary:
            x = torch.cat([x, boundaries], dim=-1)

        return x 
    
    def _initialize_weights_randomly(self):

        use_sqrt = True

        if use_sqrt:
            def f(n):
                return math.sqrt( 2.0/float(n) )
        else:
            def f(n):
                return 2.0/float(n)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                n = np.prod( m.kernel_size ) * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))

"""
    Extraction
"""

def get_model_from_json():
    model = Lipreading()
    calculateNorm2(model)
    return model

def extract_feats(model,data):
    """
    :rtype: FloatTensor
    """
    model.eval()
    crop_size = (88, 88)
    (mean, std) = (0.421, 0.165)
    preprocessor = Compose([
        Normalize( 0.0,255.0 ),
        CenterCrop(crop_size),
        Normalize(mean, std) ]
        )
    
    data = preprocessor(data)
    
    return model(torch.FloatTensor(data)[None, None, :, :, :].cuda(), lengths=[data.shape[0]])

class LipEmbedding(nn.Module):
    def __init__(self):
        super(LipEmbedding, self).__init__()

        self.model = get_model_from_json()

        path_chkpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),'extractor_lrw_resnet18_dctcn_video_boundary.pt')

        self.model.load_state_dict(torch.load(path_chkpt,map_location=torch.device('cpu')))
        self.model.eval()

    def forward(self,x):
        # x : [T, W, H]
        return  self.model(data = torch.FloatTensor(x)[None, None, :, :, :] )