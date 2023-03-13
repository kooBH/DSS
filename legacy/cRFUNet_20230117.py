import torch
import torch.nn as nn

from UNet.UNet import UNet
from UNet.UNet_m import Encoder,Decoder



class UNet10(nn.Module) :
    def __init__(self, 
                 c_in = 4+6+4,
                 c_out = 4,
                 L_t = 0,
                 L_f = 0,
                 n_target=4,
                 n_feature=12,
                 n_fft=512,
                 device="cuda:0",
                 print_shape=False,
                 mask="Softplus"
                 ):
        super().__init__()

        n_hfft = int(n_fft/2+1)

        self.print_shape=print_shape

        self.N = n_target 
        self.C = c_in
        self.c_out = c_out
        self.L_t = L_t
        self.L_f = L_f
        self.F = n_hfft
        self.T = 125

        f_dim = 32

        # Model Implementation
        encoders=[]
        encoders.append(Encoder(c_in,f_dim,(7,5),
        (2,2),(2,1)))
        encoders.append(Encoder(f_dim,f_dim*2,(7,5),
        (2,2),(0,0)))
        encoders.append(Encoder(f_dim*2,f_dim*2,(5,3),(2,2),(0,0)))
        encoders.append(Encoder(f_dim*2,f_dim*2,(5,3),(2,2),(0,0)))
        encoders.append(Encoder(f_dim*2,f_dim*2,(5,3),(2,2),(0,0)))

        decoders=[]
        decoders.append(Decoder(f_dim*2,f_dim*2,(4,5),
        (2,1),(1,0)))
        decoders.append(Decoder(f_dim*4,f_dim*2,(4,4),(2,2),(1,0)))
        decoders.append(Decoder(f_dim*4,f_dim*2,(5,3),(2,2),(0,0)))
        decoders.append(Decoder(f_dim*4,f_dim*2,(7,5),(2,2),(1,1)))
        decoders.append(Decoder(f_dim*3,f_dim*2,(6,5),
        (2,2),(1,0),padding=(2,1)))


        ## Model Construction
        self.linear = nn.Conv2d(f_dim*2, 2*c_out*n_target, 1)
        self.add_module("linear", self.linear)

        self.len_model = len(encoders)

        self.encoders = encoders
        self.decoders = decoders
        for i in range(len(encoders)) : 
            module = self.encoders[i]
            self.add_module("encoder_{}".format(i),module)
        for i in range(len(decoders)) : 
            module = self.decoders[i]
            self.add_module("decoder_{}".format(i),module)

        # Residual Path
        self.res_paths = []
        if len(self.res_paths) != 0  :
            if (len(res_paths) != self.len_model -1) :
                raise Exception("ERROR::unmatched res_path : {} != {}".format(len(res_paths),self.len_model-1))
            else :
                for i in range(len(res_paths)):
                    module = res_paths[i]
                    self.add_module("res_path{}".format(i),module)
                    self.res_paths.append(module)
        # default : skip connection
        else :
            for i in range(self.len_model-1):
                module = nn.Identity()
                self.add_module("res_path{}".format(i),module)
                self.res_paths.append(module)
        # Dummy
        module = nn.Identity()
        self.add_module("res_path{}".format(i+1),module)
        self.res_paths.append(module)
            
        # Bottlenect
        self.bottlenecks = []
        if len(self.bottlenecks) :
            for i in range(len(bottlenecks)):
                module = bottlenecks[i]
                self.add_module("bottleneck{}".format(i),module)
                self.bottlenecks.append(module)
        else :
            module = nn.Identity()
            self.add_module("bottleneck{}".format(0),module)
            self.bottlenecks.append(module)

        if mask == "Sigmoid" : 
            self.activation_mask = nn.Sigmoid() 
        else :
            self.activation_mask = nn.Softplus() 


        bottleneck_channel = encoders[-1].conv.out_channels



    def forward(self,x):
        # ipnut : [ Batch Channel Freq Time]

        # Time must be multiple of something
        """
        len_orig = x.shape[-1]
        need =  int(16*np.floor(len_orig/16)+16) - len_orig
        x = torch.nn.functional.pad(x,(0,need))
        """

        # Encoder 
        x_skip = []
        for i, encoder in enumerate(self.encoders):
            x_skip.append(self.res_paths[i](x))
            x = encoder(x)
            if self.print_shape : 
                print("Encoder {} : {}".format(i,x.shape))

        p = x
        for i, bottleneck in enumerate(self.bottlenecks):
            p  =  bottleneck(p)
            if self.print_shape : 
                print("bottleneck {} : {}".format(i,p.shape))
        
        # Decoders
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            if self.print_shape : 
                print("Decoder {} : {}".format(i,p.shape))
            # last layer of Decorders
            if i == self.len_model- 1:
                break
            p = torch.cat([p, x_skip[self.len_model - 1 - i]], dim=1)
            if self.print_shape : 
                print("Decoder cat {} : {}".format(i,p.shape))

        filter = self.linear(p)
        filter = self.activation_mask(filter)
        # => filter [B,2*C*N,F,T]

        # [B,N, C, filter, n_hfft, Time, 2(complex)]
        B = filter.shape[0]
        filter = torch.reshape(filter,(B,self.N,self.c_out, (2*self.L_f+1),(2*self.L_t+1),self.F,self.T,2))

        # Return in Complex type
        return torch.view_as_complex(filter)

class UNet20(nn.Module) :
    def __init__(self, 
                 c_in = 4+6+4,
                 c_out = 4,
                 L_t = 0,
                 L_f = 0,
                 n_target=4,
                 n_feature=12,
                 n_fft=512,
                 device="cuda:0",
                 print_shape=False,
                 mask="Softplus"
                 ):
        super().__init__()

        n_hfft = int(n_fft/2+1)

        self.print_shape=print_shape

        self.N = n_target 
        self.C = c_in
        self.c_out = c_out
        self.L_t = L_t
        self.L_f = L_f
        self.F = n_hfft
        self.T = 125

        f_dim = 32

        # Model Implementation
        encoders=[]
        encoders.append(Encoder(c_in,f_dim,(7,1),
        (1,1),(3,0)))
        encoders.append(Encoder(f_dim,f_dim*2,(1,7),
        (1,1),(0,3)))
        encoders.append(Encoder(f_dim*2,f_dim*2,(7,5),(2,2),(3,2)))
        encoders.append(Encoder(f_dim*2,f_dim*2,(7,5),(2,1),(3,2)))
        encoders.append(Encoder(f_dim*2,f_dim*2,(5,3),(2,2),(2,1)))
        encoders.append(Encoder(f_dim*2,f_dim*2,(5,3),(2,1),(2,1)))
        encoders.append(Encoder(f_dim*2,f_dim*2,(5,3),(2,2),(2,1)))
        encoders.append(Encoder(f_dim*2,f_dim*2,(5,3),(2,1),(2,1)))
        encoders.append(Encoder(f_dim*2,f_dim*2,(5,3),(2,2),(2,1)))
        encoders.append(Encoder(f_dim*2,f_dim*2,(5,3),(2,1),(2,1)))

        decoders=[]
        decoders.append(Decoder(f_dim*2,f_dim*2,(5,3),
        (2,1),padding=(2,1)))
        decoders.append(Decoder(f_dim*4,f_dim*2,(5,3),(2,2),padding=(2,1),output_padding=(0,1)))
        decoders.append(Decoder(f_dim*4,f_dim*2,(5,3),(2,1),padding=(2,1)))
        decoders.append(Decoder(f_dim*4,f_dim*2,(5,3),(2,2),padding=(2,1),output_padding=(0,1)))
        decoders.append(Decoder(f_dim*4,f_dim*2,(5,3),(2,1),padding=(2,1)))
        decoders.append(Decoder(f_dim*4,f_dim*2,(5,3),(2,2),padding=(2,1)))
        decoders.append(Decoder(f_dim*4,f_dim*2,(7,5),(2,1),padding=(3,2)))
        decoders.append(Decoder(f_dim*4,f_dim*2,(7,5),(2,2),padding=(3,2)))
        decoders.append(Decoder(f_dim*4,f_dim*2,(1,7),(1,1),padding=(0,3)))
        decoders.append(Decoder(f_dim*3,f_dim*2,(7,1),
        (1,1),padding=(3,0)))


        ## Model Construction
        self.linear = nn.Conv2d(f_dim*2, 2*c_out*n_target, 1)
        self.add_module("linear", self.linear)

        self.len_model = len(encoders)

        self.encoders = encoders
        self.decoders = decoders
        for i in range(len(encoders)) : 
            module = self.encoders[i]
            self.add_module("encoder_{}".format(i),module)
        for i in range(len(decoders)) : 
            module = self.decoders[i]
            self.add_module("decoder_{}".format(i),module)

        # Residual Path
        self.res_paths = []
        if len(self.res_paths) != 0  :
            if (len(res_paths) != self.len_model -1) :
                raise Exception("ERROR::unmatched res_path : {} != {}".format(len(res_paths),self.len_model-1))
            else :
                for i in range(len(res_paths)):
                    module = res_paths[i]
                    self.add_module("res_path{}".format(i),module)
                    self.res_paths.append(module)
        # default : skip connection
        else :
            for i in range(self.len_model-1):
                module = nn.Identity()
                self.add_module("res_path{}".format(i),module)
                self.res_paths.append(module)
        # Dummy
        module = nn.Identity()
        self.add_module("res_path{}".format(i+1),module)
        self.res_paths.append(module)
            
        # Bottlenect
        self.bottlenecks = []
        if len(self.bottlenecks) :
            for i in range(len(bottlenecks)):
                module = bottlenecks[i]
                self.add_module("bottleneck{}".format(i),module)
                self.bottlenecks.append(module)
        else :
            module = nn.Identity()
            self.add_module("bottleneck{}".format(0),module)
            self.bottlenecks.append(module)

        if mask == "Sigmoid" : 
            self.activation_mask = nn.Sigmoid() 
        else :
            self.activation_mask = nn.Softplus() 


        bottleneck_channel = encoders[-1].conv.out_channels



    def forward(self,x):
        # ipnut : [ Batch Channel Freq Time]

        # Time must be multiple of something
        """
        len_orig = x.shape[-1]
        need =  int(16*np.floor(len_orig/16)+16) - len_orig
        x = torch.nn.functional.pad(x,(0,need))
        """

        # Encoder 
        x_skip = []
        for i, encoder in enumerate(self.encoders):
            x_skip.append(self.res_paths[i](x))
            x = encoder(x)
            if self.print_shape : 
                print("Encoder {} : {}".format(i,x.shape))

        p = x
        for i, bottleneck in enumerate(self.bottlenecks):
            p  =  bottleneck(p)
            if self.print_shape : 
                print("bottleneck {} : {}".format(i,p.shape))
        
        # Decoders
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            if self.print_shape : 
                print("Decoder {} : {}".format(i,p.shape))
            # last layer of Decorders
            if i == self.len_model- 1:
                break
            p = torch.cat([p, x_skip[self.len_model - 1 - i]], dim=1)
            if self.print_shape : 
                print("Decoder cat {} : {}".format(i,p.shape))

        filter = self.linear(p)
        filter = self.activation_mask(filter)
        # => filter [B,2*C*N,F,T]

        # [B,N, C, filter, n_hfft, Time, 2(complex)]
        B = filter.shape[0]
        filter = torch.reshape(filter,(B,self.N,self.c_out, (2*self.L_f+1),(2*self.L_t+1),self.F,self.T,2))

        # Return in Complex type
        return torch.view_as_complex(filter)

if __name__ == "__main__" : 
        x = torch.rand(1,14,257,125)
        print(x.shape)

        #m = UNet10(c_in = 14,c_out = 4,n_fft = 512,print_shape=True)
        m = UNet20(c_in = 14,c_out = 4,n_fft = 512,print_shape=True)

        y = m(x)

        print(y.shape)


