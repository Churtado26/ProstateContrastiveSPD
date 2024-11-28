import torchvision.models as models
import torch
from torch import nn
from torchsummary import summary
import numpy as np
import  spdnetwork.nn as nn_spd
from spdnetwork.optimizers import  MixOptimizer 
from spdnetwork.nn import LogEig
import torch.nn as nn

#MODELS BY JAOR
def CB1():
    sequential_layers = nn.Sequential(
        #Block
        nn.Conv3d( 
                in_channels=1, 
                out_channels=8, 
                kernel_size=(1,3,3),                
                padding='same',
                dtype=torch.double                
            ), 
        nn.BatchNorm3d(8,dtype=torch.double),
        nn.ReLU(), # (b,8,D,H,W)
        nn.Conv3d( 
                in_channels=8, 
                out_channels=8, 
                kernel_size=(3,3,3),                
                padding='same',
                dtype=torch.double
            ), 
        nn.BatchNorm3d(8,dtype=torch.double),
        nn.ReLU(), # (b,8,D,H,W)
        #Maxpool
        nn.MaxPool3d(kernel_size= (1,2,2)), # (b,8,D,H/2,W/2)
        nn.Conv3d( 
                in_channels=8, 
                out_channels=16, 
                kernel_size=(1,3,3),                
                padding='same',
                dtype=torch.double                
            ),
        nn.BatchNorm3d(16,dtype=torch.double),
        nn.ReLU()  # (b,16,D,H/2,W/2)
    )
    return sequential_layers 


def CB2():
    sequential_layers = nn.Sequential(
        #Block
        nn.Conv3d( 
                in_channels=1, 
                out_channels=8, 
                kernel_size=(1,3,3),                
                padding='same',
                dtype=torch.double               
            ), 
        nn.BatchNorm3d(8,dtype=torch.double),
        nn.ReLU(),        
        nn.Conv3d( 
                in_channels=8, 
                out_channels=8, 
                kernel_size=(3,3,3),                
                padding='same',
                dtype=torch.double
            ), 
        nn.BatchNorm3d(8,dtype=torch.double),
        nn.ReLU(),
        #Maxpool
        nn.MaxPool3d(kernel_size= (1,2,2)),
        #Block
        nn.Conv3d( 
                in_channels=8, 
                out_channels=16, 
                kernel_size=(1,3,3),                
                padding='same',
                dtype=torch.double              
            ), 
        nn.BatchNorm3d(16,dtype=torch.double),
        nn.ReLU(),        
        nn.Conv3d( 
                in_channels=16, 
                out_channels=16, 
                kernel_size=(3,3,3),                
                padding='same',
                dtype=torch.double
            ), 
        nn.BatchNorm3d(16,dtype=torch.double),
        nn.ReLU(),
        #Maxpool
        nn.MaxPool3d(kernel_size= (2,2,2)),
        nn.Conv3d( 
                in_channels=16, 
                out_channels=32, 
                kernel_size=(1,3,3),                
                padding='same',
                dtype=torch.double
            ),
        nn.BatchNorm3d(32,dtype=torch.double),
        nn.ReLU(),
    )
    return sequential_layers 


def CB2_attn(in_dim, out_dim, nh):
    sequential_layers = nn.Sequential(
        #Block
        nn.Conv3d( 
                in_channels=in_dim, 
                out_channels=8*nh, 
                kernel_size=(1,3,3),                
                padding='same',
                dtype=torch.double               
            ), 
        nn.BatchNorm3d(8*nh,dtype=torch.double),
        nn.ReLU(),        
        nn.Conv3d( 
                in_channels=8*nh, 
                out_channels=8*nh, 
                kernel_size=(3,3,3),                
                padding='same',
                dtype=torch.double
            ), 
        nn.BatchNorm3d(8*nh,dtype=torch.double),
        nn.ReLU(),
        #Maxpool
        nn.MaxPool3d(kernel_size= (1,2,2)),
        #Block
        nn.Conv3d( 
                in_channels=8*nh, 
                out_channels=16*nh, 
                kernel_size=(1,3,3),                
                padding='same',
                dtype=torch.double              
            ), 
        nn.BatchNorm3d(16*nh,dtype=torch.double),
        nn.ReLU(),        
        nn.Conv3d( 
                in_channels=16*nh, 
                out_channels=16*nh, 
                kernel_size=(3,3,3),                
                padding='same',
                dtype=torch.double
            ), 
        nn.BatchNorm3d(16*nh,dtype=torch.double),
        nn.ReLU(),
        #Maxpool
        nn.MaxPool3d(kernel_size= (2,2,2)),
        nn.Conv3d( 
                in_channels=16*nh, 
                out_channels=out_dim, 
                kernel_size=(1,3,3),                
                padding='same',
                dtype=torch.double
            ),
        nn.BatchNorm3d(out_dim,dtype=torch.double),
        nn.ReLU(),
    )
    return sequential_layers 



def CB3():
    sequential_layers = nn.Sequential(
        #Blockß
        nn.Conv3d( 
                in_channels=1, 
                out_channels=8, 
                kernel_size=(1,3,3),                
                padding='same',
                dtype=torch.double               
            ), # (b,8,D,H,W)
        nn.BatchNorm3d(8,dtype=torch.double),
        nn.ReLU(),        
        nn.Conv3d( 
                in_channels=8, 
                out_channels=8, 
                kernel_size=(3,3,3),                
                padding='same',
                dtype=torch.double
            ), # (b,8,D,H,W)
        nn.BatchNorm3d(8,dtype=torch.double),
        nn.ReLU(),
        #Maxpool
        nn.MaxPool3d(kernel_size= (1,2,2)), # (b,8,D,H/2,W/2)
        #Block
        nn.Conv3d( 
                in_channels=8, 
                out_channels=16, 
                kernel_size=(1,3,3),                
                padding='same',
                dtype=torch.double              
            ), 
        nn.BatchNorm3d(16,dtype=torch.double),
        nn.ReLU(),   # (b,16,D,H/2,W/2)
        nn.Conv3d( 
                in_channels=16, 
                out_channels=16, 
                kernel_size=(3,3,3),                
                padding='same',
                dtype=torch.double
            ), 
        nn.BatchNorm3d(16,dtype=torch.double),
        nn.ReLU(), # (b,16,D,H/2,W/2)
        #Maxpool
        nn.MaxPool3d(kernel_size= (2,2,2)), # (b,16,D/2,H/4,W/4)
        nn.Conv3d( 
                in_channels=16, 
                out_channels=32, 
                kernel_size=(1,3,3),                
                padding='same',
                dtype=torch.double
            ),
        nn.BatchNorm3d(32,dtype=torch.double),
        nn.ReLU(), # (b,32,D/2,H/4,W/4)
        nn.Conv3d( 
                in_channels=32, 
                out_channels=32, 
                kernel_size=(3,3,3),                
                padding='same',
                dtype=torch.double
            ),
        nn.BatchNorm3d(32,dtype=torch.double),
        nn.ReLU(), # (b,32,D/2,H/4,W/4)
        nn.MaxPool3d(kernel_size= (1,2,2)), # (b,32,D/2,H/8,W/8)
        #Block
        nn.Conv3d( 
                in_channels=32, 
                out_channels=64, 
                kernel_size=(1,3,3),                
                padding='same',
                dtype=torch.double              
            ), 
        nn.BatchNorm3d(64,dtype=torch.double),
        nn.ReLU() # (b,64,D/2,H/8,W/8)
    )
    return sequential_layers 



class CB1_BiRe1(nn.Module):
    def __init__(self, device):
        super(CB1_BiRe1, self).__init__()  
        self.t2_conv_branch = CB1()    # CNN backbone for t2 ->(b,16,D,H/2,W/2)
        self.adc_conv_branch = CB1()   # CNN backbone for adc
        self.hbv_conv_branch = CB1()  # CNN backbone for hbv
        
        self.covariance = nn_spd.CovPool()
        self.spd_module = nn.Sequential(
            nn_spd.BiMap(1, 1, 3 * 16, 16, dtype=torch.double, device=device),  # Input: (48,48) -> (16,16)
            nn_spd.ReEig(),
            nn_spd.LogEig()
        )
        self.linear = torch.nn.Linear(16 * (16+1) //2, 1, dtype=torch.double)  # (n,n) -- UT flatten --> n * (n+1) /2

        # Precompute the upper triangular indices
        self.indices = torch.triu_indices(16, 16) #Upper triangular part is filled with 1s.
        
    def forward(self, X):        
        t2_img = X[:, 0, ...].unsqueeze(1)  # add channel dimension        
        adc_img = X[:, 1, ...].unsqueeze(1)        
        hbv_img = X[:, 2, ...].unsqueeze(1)
        
        batch_size = adc_img.shape[0]
      
        # Convolutional embeddings
        t2_embedding = self.t2_conv_branch(t2_img).view(batch_size, 16, -1) # (b,16,D,H/2,W/2) ->(b, 16, d*h/2*w/2)
        adc_embedding = self.adc_conv_branch(adc_img).view(batch_size, 16, -1)  # (b,16,D,H/2,W/2) ->(b, 16, d*h/2*w/2)
        hbv_embedding = self.hbv_conv_branch(hbv_img).view(batch_size, 16, -1) # (b,16,D,H/2,W/2) ->(b, 16, d*h/2*w/2)
        
        fusion = torch.cat((t2_embedding, adc_embedding, hbv_embedding ), dim=1)  # (b, 3*16, d*h*w)
        
        # Covariance and SPD
        covariance = self.covariance(fusion)  # (b, 48, 48)
        spd_output = self.spd_module(covariance).squeeze(1)  # (b, 16, 16)
        
        # Upper triangular extraction using precomputed indices
        upper_triangular = spd_output[:, self.indices[0], self.indices[1]]  # (batch, 136) upper triangular part of the matrix
        
        # Linear layer
        pred = self.linear(upper_triangular)  # (batch, 1)
        
        return pred 



class CB1_BiRe1_int1(nn.Module):
    def __init__(self, device):
        super(CB1_BiRe1_int1, self).__init__()  
        self.t2_conv_branch = CB1()   # CNN backbone for t2
        self.adc_conv_branch = CB1()  # CNN backbone for adc
        self.hbv_conv_branch = CB1()  # CNN backbone for hbv
        self.covariance_t2 = nn_spd.CovPool()
        self.covariance_adc = nn_spd.CovPool()
        self.covariance_hbv = nn_spd.CovPool()
        self.spd_module = nn.Sequential(
            nn_spd.BiMap(1, 3, 16, 8, dtype=torch.double, device=device),  # (No.Mat output, No.matrices input, Size input, size output)
            nn_spd.ReEig(),
            nn_spd.LogEig()
        )
        self.linear = torch.nn.Linear(8 * (8 + 1) // 2, 1, dtype=torch.double)  # Adjust the output size

        # Precompute the upper triangular indices
        self.indices = torch.triu_indices(8, 8)
        
    def forward(self, X):        
        t2_img = X[:, 0, ...].unsqueeze(1)  # add channel dimension        
        adc_img = X[:, 1, ...].unsqueeze(1)        
        hbv_img = X[:, 2, ...].unsqueeze(1)
        
        batch_size = adc_img.shape[0]
      
        # Convolutional embeddings in the desired order
        t2_embedding = self.t2_conv_branch(t2_img).view(batch_size, 16, -1)  # (batch, 16, d*h*w)
        adc_embedding = self.adc_conv_branch(adc_img).view(batch_size, 16, -1)
        hbv_embedding = self.hbv_conv_branch(hbv_img).view(batch_size, 16, -1)

        # Covariance matrices
        spd_t2 = self.covariance_t2(t2_embedding)
        spd_adc = self.covariance_adc(adc_embedding)
        spd_hbv = self.covariance_hbv(hbv_embedding)

        fusion = torch.cat((spd_t2, spd_adc, spd_hbv), dim=1)        
        # SPD module
        spd_output = self.spd_module(fusion).squeeze(1)

        # Extract and flatten the upper triangular part using precomputed indices
        upper_triangular = spd_output[:, self.indices[0], self.indices[1]]
        # print(f'Upper triangular shape: {upper_triangular.shape}')
        pred = self.linear(upper_triangular)  # (batch, 1)

        return pred



class CB1_noSPD(nn.Module):
    def __init__(self, device):
        super(__class__,self).__init__()          
        self.t2_conv_branch =  CB1()
        self.adc_conv_branch =  CB1()
        self.hbv_conv_branch =  CB1()
        
        self.t2_gap = nn.AdaptiveAvgPool3d((3,1,1)) #output shape (bs, 16, 3, 1, 1)
        self.adc_gap = nn.AdaptiveAvgPool3d((3,1,1))
        self.hbv_gap = nn.AdaptiveAvgPool3d((3,1,1))

        self.linear_fusion1 = torch.nn.Linear(3*16*3, 1, dtype=torch.double)        

    def forward(self, X):
        t2_img = X[:,0,...].unsqueeze(1) #add channel dimension        
        adc_img = X[:,1,...].unsqueeze(1)
        hbv_img = X[:,2,...].unsqueeze(1)
        
        batch_size = adc_img.shape[0]
        
        t2_embedding = self.t2_conv_branch(t2_img)
        # print(f'T2 shape after conv: {t2_embedding.shape}')
        t2_embedding = self.t2_gap(t2_embedding).view(batch_size,-1)
        # print(f'T2 shape after gap: {t2_embedding.shape}')
        
        adc_embedding = self.adc_conv_branch(adc_img)
        # print(f'ADC shape after conv: {adc_embedding.shape}')
        adc_embedding = self.adc_gap(adc_embedding).view(batch_size,-1)
        # print(f'ADC shape after gap: {adc_embedding.shape}')
        
        hbv_embedding = self.hbv_conv_branch(hbv_img)
        # print(f'hbv shape after conv: {hbv_embedding.shape}')
        hbv_embedding = self.hbv_gap(hbv_embedding).view(batch_size,-1)
        # print(f'hbv shape after gap: {hbv_embedding.shape}')

        fusion = torch.cat((t2_embedding,adc_embedding, hbv_embedding ), dim = 1)
        # print(f'Fusion shape: {fusion.shape}')
        fusion = self.linear_fusion1(fusion)
        # print(f'Fusion shape: {fusion.shape}')
        return fusion
    


class CB1_BiRe1_int2(nn.Module):
    def __init__(self, device):
        super(__class__,self).__init__()          
        self.t2_conv_branch =  CB1()
        self.adc_conv_branch =  CB1()
        self.hbv_conv_branch =  CB1() #CNN backbone... hbv_conv_branch()
        
        self.covariance_adc = nn_spd.CovPool()
        self.covariance_t2 = nn_spd.CovPool()
        self.covariance_hbv  = nn_spd.CovPool()
        self.hbv_spd_module  = nn.Sequential(
                    nn_spd.BiMap(1 , 1, 16, 8, dtype=torch.double, device = device), #(No.Mat output,No.matrices input,Size input,size output)
                    nn_spd.ReEig(),
                    nn_spd.LogEig()
                )
        self.adc_spd_module  = nn.Sequential(
                    nn_spd.BiMap(1, 1, 16, 8, dtype=torch.double, device = device), #(No.Mat output,No.matrices input,Size input,size output)
                    nn_spd.ReEig(),
                    nn_spd.LogEig()
                )
        self.t2_spd_module = nn.Sequential(
                    nn_spd.BiMap(1, 1, 16, 8, dtype=torch.double, device = device), #(No.Mat output,No.matrices input,Size input,size output)
                    nn_spd.ReEig(),
                    nn_spd.LogEig()
                )
        self.linear = torch.nn.Linear(3 * 8 * (8+1) //2, 1, dtype=torch.double)
        
        # Precompute the upper triangular indices
        self.indices = torch.triu_indices(8, 8)
        
    def forward(self, X):        
        t2_img = X[:,0,...].unsqueeze(1) #add channel dimension        
        adc_img = X[:,1,...].unsqueeze(1)        
        hbv_img = X[:,2,...].unsqueeze(1)
        # print(f'Input shape t2: {t2_img.shape}')
        batch_size = adc_img.shape[0]
      
        t2_embedding = self.t2_conv_branch(t2_img).view(batch_size, 16, -1)
        adc_embedding = self.adc_conv_branch(adc_img).view(batch_size, 16, -1) #(batch,CH,D,H,W)...(b,16,d,h,w) el view (batch,16,d*h*w)
        hbv_embedding = self.hbv_conv_branch(hbv_img).view(batch_size, 16, -1)
        
        spd_t2 = self.covariance_t2(t2_embedding)
        spd_adc = self.covariance_adc(adc_embedding)
        spd_hbv = self.covariance_hbv(hbv_embedding)

        t2_log_spd = self.t2_spd_module(spd_t2).squeeze(1)
        adc_log_spd = self.adc_spd_module(spd_adc).squeeze(1)
        hbv_log_spd = self.hbv_spd_module(spd_hbv).squeeze(1)
        # print(f't2_log_spd: {t2_log_spd.shape}')
        t2_upper_triangular = t2_log_spd[:, self.indices[0], self.indices[1]]
        adc_upper_triangular = adc_log_spd[:, self.indices[0], self.indices[1]]
        hbv_upper_triangular = hbv_log_spd[:, self.indices[0], self.indices[1]]
        # print(f't2_upper_triangular: {t2_upper_triangular.shape}')
        fusion = torch.cat((t2_upper_triangular, adc_upper_triangular, hbv_upper_triangular ), dim = 1)        
        # print(f'Fusion shape: {fusion.shape}')
     
        pred = self.linear(fusion) 
        return pred



class CB2_BiRe1(nn.Module):
    def __init__(self, device):
        super(__class__,self).__init__()                  
        self.t2_conv_branch =  CB2()
        self.adc_conv_branch =  CB2()
        self.hbv_conv_branch =  CB2()
        self.covariance  = nn_spd.CovPool()
        self.spd_module  = nn.Sequential(
            nn_spd.BiMap(1 , 1, 3*32, 32, dtype=torch.double, device = device), #(No.Mat output,No.matrices input,Size input,size output)
            nn_spd.ReEig(),
            nn_spd.LogEig()
        )
        self.linear = torch.nn.Linear(32 * (32+1) //2, 1, dtype=torch.double)
        
        # Precompute the upper triangular indices
        self.indices = torch.triu_indices(32, 32)
    def forward(self, X):        
        t2_img = X[:,0,...].unsqueeze(1) #add channel dimension        
        adc_img = X[:,1,...].unsqueeze(1)        
        hbv_img = X[:,2,...].unsqueeze(1)        
        batch_size = adc_img.shape[0]
      
        t2_embedding = self.t2_conv_branch(t2_img).view(batch_size, 32, -1)
        adc_embedding = self.adc_conv_branch(adc_img).view(batch_size, 32, -1)                
        hbv_embedding = self.hbv_conv_branch(hbv_img).view(batch_size, 32, -1)
        
        fusion = torch.cat((t2_embedding, adc_embedding, hbv_embedding), dim = 1)
        covariance = self.covariance(fusion)
        spd_output = self.spd_module(covariance).squeeze(1)
        
        upper_triangular = spd_output[:, self.indices[0], self.indices[1]]  # (batch, 136)        
        pred = self.linear(upper_triangular)
        return pred
    



class CB2_noSPD(nn.Module):
    def __init__(self, device):
        super(__class__,self).__init__()          
        self.t2_conv_branch =  CB2()
        self.adc_conv_branch =  CB2()
        self.hbv_conv_branch =  CB2()
        
        self.t2_gap = nn.AdaptiveAvgPool3d((3,1,1)) #output shape (bs, 16, 3, 1, 1)
        self.adc_gap = nn.AdaptiveAvgPool3d((3,1,1))
        self.hbv_gap = nn.AdaptiveAvgPool3d((3,1,1))

        self.linear_fusion1 = torch.nn.Linear(3*32*3, 1, dtype=torch.double)        

    def forward(self, X):
        # print(f'Input shape: {X.shape} dtype: {X.dtype} device: {X.device}')
        t2_img = X[:,0,...].unsqueeze(1) #add channel dimension        
        adc_img = X[:,1,...].unsqueeze(1)
        hbv_img = X[:,2,...].unsqueeze(1)        
        batch_size = adc_img.shape[0]
        
        
        t2_embedding = self.t2_conv_branch(t2_img)        
        t2_embedding = self.t2_gap(t2_embedding).view(batch_size,-1)
        
        adc_embedding = self.adc_conv_branch(adc_img)
        adc_embedding = self.adc_gap(adc_embedding).view(batch_size,-1)
        
        hbv_embedding = self.hbv_conv_branch(hbv_img)
        hbv_embedding = self.hbv_gap(hbv_embedding).view(batch_size,-1)

        fusion = torch.cat((t2_embedding,adc_embedding, hbv_embedding ), dim = 1)
        # print(f'Fusion shape: {fusion.shape}')
        fusion = self.linear_fusion1(fusion)
        # print(f'Fusion shape: {fusion.shape}')
        return fusion
    



class CB2_BiRe1_int1(nn.Module):
    def __init__(self, device):
        super(__class__,self).__init__()  
        self.t2_conv_branch =  CB2()
        self.adc_conv_branch =  CB2()
        self.hbv_conv_branch =  CB2() #CNN backbone... hbv_conv_branch()
    
        self.covariance_adc = nn_spd.CovPool()
        self.covariance_t2 = nn_spd.CovPool()
        self.covariance_hbv  = nn_spd.CovPool()
        self.spd_module  = nn.Sequential(
            nn_spd.BiMap(1 , 3, 32, 16, dtype=torch.double, device = device), #(No.Mat output,No.matrices input,Size input,size output)
            nn_spd.ReEig(),
            nn_spd.LogEig()
        )
        self.linear = torch.nn.Linear(16*17//2, 1, dtype=torch.double)
        # Precompute the upper triangular indices
        self.indices = torch.triu_indices(16, 16)
        
    def forward(self, X):        
        t2_img = X[:,0,...].unsqueeze(1) #add channel dimension        
        adc_img = X[:,1,...].unsqueeze(1)        
        hbv_img = X[:,2,...].unsqueeze(1)        
        batch_size = adc_img.shape[0]
      
        t2_embedding = self.t2_conv_branch(t2_img).view(batch_size, 32, -1)
        adc_embedding = self.adc_conv_branch(adc_img).view(batch_size, 32, -1) #(batch,CH,D,H,W)...(b,16,d,h,w) el view (batch,16,d*h*w)
        hbv_embedding = self.hbv_conv_branch(hbv_img).view(batch_size, 32, -1)
        
        spd_t2 = self.covariance_t2(t2_embedding)
        spd_adc = self.covariance_adc(adc_embedding)
        spd_hbv = self.covariance_hbv(hbv_embedding)

        fusion = torch.cat((spd_t2, spd_adc, spd_hbv ), dim = 1)
        # print(f'Fusion shape: {fusion.shape}')
        spd_output = self.spd_module(fusion).squeeze(1)
        # print(f'spd_output shape: {spd_output.shape}')
        upper_triangular = spd_output[:, self.indices[0], self.indices[1]]          
        # print(f'upper_triangular shape: {upper_triangular.shape}')
        pred = self.linear(upper_triangular) 
        return pred 

class CB2_BiRe1_int2(nn.Module):
    def __init__(self, device):
        super(__class__,self).__init__()          
        self.adc_conv_branch =  CB2()
        self.t2_conv_branch =  CB2()
        self.hbv_conv_branch =  CB2() #CNN backbone... hbv_conv_branch()
        
        self.covariance_t2 = nn_spd.CovPool()
        self.covariance_hbv  = nn_spd.CovPool()
        self.covariance_adc = nn_spd.CovPool()
                
        self.adc_spd_module  = nn.Sequential(
                    nn_spd.BiMap(1, 1, 32, 16, dtype=torch.double, device = device), #(No.Mat output,No.matrices input,Size input,size output)
                    nn_spd.ReEig(),
                    nn_spd.LogEig()
                )
        self.t2_spd_module = nn.Sequential(
                    nn_spd.BiMap(1, 1, 32, 16, dtype=torch.double, device = device), #(No.Mat output,No.matrices input,Size input,size output)
                    nn_spd.ReEig(),
                    nn_spd.LogEig()
                )
        self.hbv_spd_module  = nn.Sequential(
                    nn_spd.BiMap(1 , 1, 32, 16, dtype=torch.double, device = device), #(No.Mat output,No.matrices input,Size input,size output)
                    nn_spd.ReEig(),
                    nn_spd.LogEig()
                )
        self.linear = torch.nn.Linear(3*16*17//2, 1, dtype=torch.double)
        
        # Precompute the upper triangular indices
        self.indices = torch.triu_indices(16, 16)
    def forward(self, X):        
        t2_img = X[:,0,...].unsqueeze(1) #add channel dimension        
        adc_img = X[:,1,...].unsqueeze(1)        
        hbv_img = X[:,2,...].unsqueeze(1)
        batch_size = adc_img.shape[0]
      
        t2_embedding = self.t2_conv_branch(t2_img).view(batch_size, 32, -1)
        adc_embedding = self.adc_conv_branch(adc_img).view(batch_size, 32, -1) #(batch,CH,D,H,W)...(b,16,d,h,w) el view (batch,16,d*h*w)
        hbv_embedding = self.hbv_conv_branch(hbv_img).view(batch_size, 32, -1)
        
        spd_t2 = self.covariance_t2(t2_embedding)
        spd_adc = self.covariance_adc(adc_embedding)
        spd_hbv = self.covariance_hbv(hbv_embedding)

        t2_log_spd = self.t2_spd_module(spd_t2).squeeze(1)
        adc_log_spd = self.adc_spd_module(spd_adc).squeeze(1)
        hbv_log_spd = self.hbv_spd_module(spd_hbv).squeeze(1)
        # print(f't2_log_spd: {t2_log_spd.shape}')
        t2_upper_triangular = t2_log_spd[:, self.indices[0], self.indices[1]]
        adc_upper_triangular = adc_log_spd[:, self.indices[0], self.indices[1]]
        hbv_upper_triangular = hbv_log_spd[:, self.indices[0], self.indices[1]]
        
        fusion = torch.cat((t2_upper_triangular, adc_upper_triangular, hbv_upper_triangular ), dim = 1)         
        # print(f'Fusion shape: {fusion.shape}')
     
        pred = self.linear(fusion) 
        return pred      
    
class CB3_BiRe1(nn.Module):
    def __init__(self, device):
        super(CB3_BiRe1, self).__init__()        
        self.adc_conv_branch = CB3()  # (b,64,D/2,H/8,W/8)
        self.t2_conv_branch = CB3()
        self.hbv_conv_branch = CB3()
        
        self.covariance = nn_spd.CovPool()
        
        self.spd_module = nn.Sequential(
            nn_spd.BiMap(1, 1, 3 * 64, 64, dtype=torch.double, device=device),
            nn_spd.ReEig(),
            nn_spd.LogEig()
        )
        self.linear = torch.nn.Linear(64*65//2, 1, dtype=torch.double) # 2080
        # Precompute the upper triangular indices
        self.indices = torch.triu_indices(64, 64)
    def forward(self, X):
        t2_img = X[:, 0, ...].unsqueeze(1) 
        adc_img = X[:, 1, ...].unsqueeze(1)
        hbv_img = X[:, 2, ...].unsqueeze(1)
        batch_size = adc_img.shape[0]

        t2_embedding = self.t2_conv_branch(t2_img).view(batch_size, 64, -1)
        adc_embedding = self.adc_conv_branch(adc_img).view(batch_size, 64, -1)
        hbv_embedding = self.hbv_conv_branch(hbv_img).view(batch_size, 64, -1)
        # print(f'adc_embedding: {adc_embedding.shape}')

        fusion = torch.cat((t2_embedding, adc_embedding, hbv_embedding), dim=1)
        covariance = self.covariance(fusion)
        # print(f'covariance: {covariance.shape}')
        spd_output = self.spd_module(covariance).squeeze(1)
        
        upper_triangular = spd_output[:, self.indices[0], self.indices[1]]         
        # print(f'upper_triangular: {upper_triangular.shape}')
        pred = self.linear(upper_triangular)
        return pred


class CB3_noSPD(nn.Module):
    def __init__(self, device):
        super(CB3_noSPD, self).__init__()
        self.t2_conv_branch = CB3()  # (b,64,D/2,H/8,W/8)
        self.adc_conv_branch = CB3()
        self.hbv_conv_branch = CB3()  
        self.t2_gap = nn.AdaptiveAvgPool3d((3, 1, 1))  # (b,64,D/2,H/8,W/8) -> (b,64,3,1,1)
        self.adc_gap = nn.AdaptiveAvgPool3d((3, 1, 1))
        self.hbv_gap = nn.AdaptiveAvgPool3d((3, 1, 1))
        self.linear_fusion1 = torch.nn.Linear(3 * 64 * 3, 1, dtype=torch.double)

    def forward(self, X):
        t2_img = X[:, 0, ...].unsqueeze(1) 
        adc_img = X[:, 1, ...].unsqueeze(1)
        hbv_img = X[:, 2, ...].unsqueeze(1)
        batch_size = adc_img.shape[0]

        t2_embedding = self.t2_conv_branch(t2_img)  # (b,64,D/2,H/8,W/8)
        t2_embedding = self.t2_gap(t2_embedding).view(batch_size, -1) #(b,64,3,1,1) -> (b,64*3) = (b,172)

        adc_embedding = self.adc_conv_branch(adc_img)
        adc_embedding = self.adc_gap(adc_embedding).view(batch_size, -1)

        hbv_embedding = self.hbv_conv_branch(hbv_img)
        hbv_embedding = self.hbv_gap(hbv_embedding).view(batch_size, -1)

        fusion = torch.cat((t2_embedding, adc_embedding, hbv_embedding), dim=1) # (b,172*3) = (b,516)
        fusion = self.linear_fusion1(fusion)
        return fusion


class CB3_BiRe1_int1(nn.Module):
    def __init__(self, device):
        super(CB3_BiRe1_int1, self).__init__()
        self.t2_conv_branch = CB3()
        self.hbv_conv_branch = CB3()
        self.adc_conv_branch = CB3()
            
        self.covariance_adc = nn_spd.CovPool()
        self.covariance_t2 = nn_spd.CovPool()
        self.covariance_hbv = nn_spd.CovPool()
        self.spd_module = nn.Sequential(
            nn_spd.BiMap(1, 3, 64, 32, dtype=torch.double, device=device),
            nn_spd.ReEig(),
            nn_spd.LogEig()
        )
        self.linear = torch.nn.Linear(32*33//2, 1, dtype=torch.double)
        # Precompute the upper triangular indices
        self.indices = torch.triu_indices(32, 32)

    def forward(self, X):
        t2_img = X[:, 0, ...].unsqueeze(1)
        adc_img = X[:, 1, ...].unsqueeze(1)
        hbv_img = X[:, 2, ...].unsqueeze(1)
        batch_size = adc_img.shape[0]

        t2_embedding = self.t2_conv_branch(t2_img).view(batch_size, 64, -1)
        adc_embedding = self.adc_conv_branch(adc_img).view(batch_size, 64, -1)
        hbv_embedding = self.hbv_conv_branch(hbv_img).view(batch_size, 64, -1)        

        spd_t2 = self.covariance_t2(t2_embedding)
        spd_adc = self.covariance_adc(adc_embedding)
        spd_hbv = self.covariance_hbv(hbv_embedding)
        # print(f'spd_t2: {spd_t2.shape}')

        fusion = torch.cat((spd_t2, spd_adc, spd_hbv), dim=1)
        # print(f'Fusion shape: {fusion.shape}')
        spd_output = self.spd_module(fusion).squeeze(1)
        upper_triangular = spd_output[:, self.indices[0], self.indices[1]]
        # print(f'upper_triangular shape: {upper_triangular.shape}')
        pred = self.linear(upper_triangular)
        return pred


class CB3_BiRe1_int2(nn.Module):
    def __init__(self, device):
        super(CB3_BiRe1_int2, self).__init__()
        self.adc_conv_branch = CB3()
        self.t2_conv_branch = CB3()
        self.hbv_conv_branch = CB3()
        
        self.covariance_adc = nn_spd.CovPool()
        self.covariance_t2 = nn_spd.CovPool()
        self.covariance_hbv = nn_spd.CovPool()
        
        self.hbv_spd_module = nn.Sequential(
            nn_spd.BiMap(1, 1, 64, 32, dtype=torch.double, device=device),
            nn_spd.ReEig(),
            nn_spd.LogEig()
        )
        self.adc_spd_module = nn.Sequential(
            nn_spd.BiMap(1, 1, 64, 32, dtype=torch.double, device=device),
            nn_spd.ReEig(),
            nn_spd.LogEig()
        )
        self.t2_spd_module = nn.Sequential(
            nn_spd.BiMap(1, 1, 64, 32, dtype=torch.double, device=device),
            nn_spd.ReEig(),
            nn_spd.LogEig()
        )
        self.linear = torch.nn.Linear(3 * 32*33//2, 1, dtype=torch.double)
        # Precompute the upper triangular indices
        self.indices = torch.triu_indices(32, 32)
    def forward(self, X):
        t2_img = X[:, 0, ...].unsqueeze(1)
        adc_img = X[:, 1, ...].unsqueeze(1)
        hbv_img = X[:, 2, ...].unsqueeze(1)
        batch_size = adc_img.shape[0]

        t2_embedding = self.t2_conv_branch(t2_img).view(batch_size, 64, -1)
        adc_embedding = self.adc_conv_branch(adc_img).view(batch_size, 64, -1) #(batch,CH,D,H,W)...(b,16,d,h,w) el view (batch,16,d*h*w)
        hbv_embedding = self.hbv_conv_branch(hbv_img).view(batch_size, 64, -1)        

        spd_t2 = self.covariance_t2(t2_embedding)
        spd_adc = self.covariance_adc(adc_embedding)
        spd_hbv = self.covariance_hbv(hbv_embedding)

        t2_log_spd = self.t2_spd_module(spd_t2).squeeze(1)
        adc_log_spd = self.adc_spd_module(spd_adc).squeeze(1)
        hbv_log_spd = self.hbv_spd_module(spd_hbv).squeeze(1)
        # print(f't2_log_spd: {t2_log_spd.shape}')
        t2_upper_triangular = t2_log_spd[:, self.indices[0], self.indices[1]]
        adc_upper_triangular = adc_log_spd[:, self.indices[0], self.indices[1]]
        hbv_upper_triangular = hbv_log_spd[:, self.indices[0], self.indices[1]]
        # print(f't2_upper_triangular: {t2_upper_triangular.shape}')
        
        fusion = torch.cat((t2_upper_triangular, adc_upper_triangular, hbv_upper_triangular ), dim = 1)         
        # print(f'Fusion shape: {fusion.shape}')
     
        pred = self.linear(fusion) 
        return pred


    


class Contrastive_CB3_SPD_1Bire(nn.Module): #Es la de yesid + SPD
    def __init__(self, device, target_shape, sequence_embedding_features=64, mode='contrastive'):
        super(Contrastive_CB3_SPD_1Bire, self).__init__()
        self._target_shape_ = target_shape
        self._sequence_embedding_features_ = sequence_embedding_features
        self._mode_ = mode
        self.t2_conv_branch =  CB3()
        self.adc_conv_branch =  CB3()
        self.bval_conv_branch =  CB3()
        self.fc_block = self.__fc_block__(mode)

        #SPD NETWORK
        self.covariance  = nn_spd.CovPool()
        self.spd_module = nn.Sequential(
            nn_spd.BiMap(1 , 1, 3*64, 64, dtype = torch.double, device = device), #(No.Mat output,No.matrices input,Size input,size output)
            nn_spd.ReEig(),
            nn_spd.LogEig()
        )
        self.indices = torch.triu_indices(64,64)
        
    
    def __fusion_block__(self, adc_embedding, bval_embedding, ktrans_embedding, zone_embedding):
        fusion_embedding = torch.cat([adc_embedding, bval_embedding, ktrans_embedding, zone_embedding], dim=1)
        return fusion_embedding
        
    def __fc_block__(self, mode = 'classifier', dropout_rate = 0.25):
        if mode == 'classifier':
            sequential_layers = nn.Sequential(
                nn.Linear(
                    in_features = 3*self._sequence_embedding_features_, 
                    out_features=3*self._sequence_embedding_features_
                ),
                nn.BatchNorm1d(3*self._sequence_embedding_features_),
                nn.Dropout(p=dropout_rate),
                nn.LeakyReLU(0.3),
                nn.Linear(
                    in_features = 3*self._sequence_embedding_features_, 
                    out_features= 97
                ),
                nn.BatchNorm1d(97),
                nn.Dropout(p=dropout_rate),
                nn.LeakyReLU(0.3),
                 nn.Linear(
                    in_features = 97, 
                    out_features=1
                ),
                nn.Sigmoid()
            ) 
            return sequential_layers
        elif mode == 'contrastive':
            #print("Entrò")
            if self._sequence_embedding_features_  <= 256:
                sequential_layers = nn.Sequential(
                         nn.Linear(
                            in_features = self._sequence_embedding_features_ , 
                            out_features=self._sequence_embedding_features_
                        ),
                        #nn.BatchNorm1d(3*self._sequence_embedding_features_ + 3),
                        #nn.Dropout(p=dropout_rate),
                        nn.LeakyReLU(0.3),
                        nn.Linear(
                            in_features = self._sequence_embedding_features_ , 
                            out_features=128
                        )
                       
                ) 
                return sequential_layers
            else:
                #print("Entrò")
                sequential_layers = nn.Sequential(
                         nn.Linear(
                            in_features = self._sequence_embedding_features_, 
                            out_features=self._sequence_embedding_features_
                        ),
                        #nn.BatchNorm1d(3*self._sequence_embedding_features_ + 3),
                        #nn.Dropout(p=dropout_rate),
                        nn.LeakyReLU(0.3),
                        nn.Linear(
                            in_features = self._sequence_embedding_features_, 
                            out_features= 256
                        ),
                        #nn.BatchNorm1d(256),
                        #nn.Dropout(p=dropout_rate),
                        nn.LeakyReLU(0.3),
                        nn.Linear(
                            in_features = 256, 
                            out_features=128
                        )
                        
                ) 
                return sequential_layers
    
    def forward(self, x):
        t2 = x[0].double()
        adc = x[1].double()
        bval = x[2].double()
        
        t2 = t2.unsqueeze(1)
        #print(f'Input shape t2: {t2.shape}')
        adc = adc.unsqueeze(1)
        bval = bval.unsqueeze(1)

        
        t2_embedding = self.t2_conv_branch(t2)
        #print(f'shape t2_embedding apply backbone: {t2_embedding.shape}')
        
        adc_embedding = self.adc_conv_branch(adc)
        bval_embedding = self.bval_conv_branch(bval)
        fusion_embedding = torch.cat([t2_embedding, adc_embedding, bval_embedding], dim=1)
        #print(f'Fusion shape: {fusion_embedding.shape}')
        fusion_embedding = fusion_embedding.view(fusion_embedding.shape[0],fusion_embedding.shape[1],-1)
        #print(f'Fusion shape: {fusion_embedding.shape}')
        
        #SPD NETWORK
        covariance = self.covariance(fusion_embedding) # hace w * w.T
        #print(f'Covariance shape: {covariance.shape}')
        
        spd_output = self.spd_module(covariance).squeeze(1) #(batch, 16,16)
        #print(f'SPD shape: {spd_output.shape}') 
        
        upper_triangular = spd_output[:, self.indices[0], self.indices[1]] #.float()
        #print(f'upper SPD shape: {upper_triangular.shape}') 
        
        logits = self.fc_block(upper_triangular)
        #print(f'logits shape: {logits.shape}')
        
        return logits
    
class Contrastive_CB3_SPD_WithOutBire(nn.Module): #Es la de yesid + SPD Pero sin el BIRE
    def __init__(self, device, target_shape, sequence_embedding_features=64, mode='classifier'):
        super(Contrastive_CB3_SPD_WithOutBire, self).__init__()
        self._target_shape_ = target_shape
        self._sequence_embedding_features_ = sequence_embedding_features
        self._mode_ = mode
        self.t2_conv_branch =  CB3()
        self.adc_conv_branch =  CB3()
        self.bval_conv_branch =  CB3()
        self.fc_block = self.__fc_block__(mode)

        #SPD NETWORK
        self.covariance  = nn_spd.CovPool()
        self.spd_module = nn.Sequential(
            nn_spd.ReEig(),
            nn_spd.LogEig()
        )
        self.indices = torch.triu_indices(192,192)
        
        
    def __fc_block__(self, mode = 'classifier', dropout_rate = 0.25):
        if mode == 'classifier':
            sequential_layers = nn.Sequential(
                nn.Linear(
                    in_features = 3*self._sequence_embedding_features_, 
                    out_features=3*self._sequence_embedding_features_
                ),
                nn.BatchNorm1d(3*self._sequence_embedding_features_),
                nn.Dropout(p=dropout_rate),
                nn.LeakyReLU(0.3),
                nn.Linear(
                    in_features = 3*self._sequence_embedding_features_, 
                    out_features= 97
                ),
                nn.BatchNorm1d(97),
                nn.Dropout(p=dropout_rate),
                nn.LeakyReLU(0.3),
                 nn.Linear(
                    in_features = 97, 
                    out_features=1
                ),
                nn.Sigmoid()
            ) 
            return sequential_layers
        elif mode == 'contrastive':
            if self._sequence_embedding_features_  <= 256:
                sequential_layers = nn.Sequential(
                         nn.Linear(
                            in_features = self._sequence_embedding_features_ , 
                            out_features=self._sequence_embedding_features_
                        ),
                        #nn.BatchNorm1d(3*self._sequence_embedding_features_ + 3),
                        #nn.Dropout(p=dropout_rate),
                        nn.LeakyReLU(0.3),
                        nn.Linear(
                            in_features = self._sequence_embedding_features_ , 
                            out_features=128
                        )
                       
                ) 
                return sequential_layers
            else:
                sequential_layers = nn.Sequential(
                         nn.Linear(
                            in_features = self._sequence_embedding_features_, 
                            out_features=512
                        ),
                        #nn.BatchNorm1d(3*self._sequence_embedding_features_ + 3),
                        #nn.Dropout(p=dropout_rate),
                        nn.LeakyReLU(0.3),
                        nn.Linear(
                            in_features = 512, 
                            out_features= 128
                        )
                ) 
                return sequential_layers
    
    def forward(self, x):
        t2 = x[0].double()
        adc = x[1].double()
        bval = x[2].double()
        
        t2 = t2.unsqueeze(1)
        #print(f'Input shape t2: {t2.shape}')
        adc = adc.unsqueeze(1)
        bval = bval.unsqueeze(1)

        
        t2_embedding = self.t2_conv_branch(t2)
        #print(f'shape t2_embedding apply backbone: {t2_embedding.shape}')
        
        adc_embedding = self.adc_conv_branch(adc)
        bval_embedding = self.bval_conv_branch(bval)
        fusion_embedding = torch.cat([t2_embedding, adc_embedding, bval_embedding], dim=1)
        #print(f'Fusion shape: {fusion_embedding.shape}')
        fusion_embedding = fusion_embedding.view(fusion_embedding.shape[0],fusion_embedding.shape[1],-1)
        #print(f'Fusion shape: {fusion_embedding.shape}')
        
        #SPD NETWORK
        covariance = self.covariance(fusion_embedding) # hace w * w.T
        #print(f'Covariance shape: {covariance.shape}')
        
        
        spd_output = self.spd_module(covariance).squeeze(1)
        #print(f'SPD output: {spd_output.shape}') 

        upper_triangular = spd_output[:, self.indices[0], self.indices[1]]
        #print(f'upper_triangular: {upper_triangular.shape}')
        
        logits = self.fc_block(upper_triangular)
        #print(f'logits shape: {logits.shape}')
        return logits

    


class MertashBiParametricNetworkV3(nn.Module): #Es la de yesid pura
    def __init__(self, device, target_shape, sequence_embedding_features=64, mode='classifier'):
        super(MertashBiParametricNetworkV3, self).__init__()
        self._target_shape_ = target_shape
        self._sequence_embedding_features_ = sequence_embedding_features
        self._mode_ = mode
        self.t2_conv_branch =  self.__conv3d_block__()
        self.adc_conv_branch =  self.__conv3d_block__()
        self.bval_conv_branch =  self.__conv3d_block__()
        self.fc_block = self.__fc_block__(mode)
        #SPD NETWORK
        self.covariance  = nn_spd.CovPool()
        self.spd_module = nn.Sequential(
            nn_spd.BiMap(1 , 1, 3*32, 16, dtype = torch.double, device = device), #(No.Mat output,No.matrices input,Size input,size output)
            nn_spd.ReEig(),
            nn_spd.LogEig()
        )
        
    
    def __conv3d_block__(self, dropout_rate = 0.25):
        sequential_layers = nn.Sequential(
            nn.Conv3d( 
                in_channels=1, 
                out_channels=4, 
                kernel_size=(1,3,3),
                padding='same',
                dtype=torch.double
            ),
            nn.BatchNorm3d(4, dtype=torch.double),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(0.3),
            
            nn.Conv3d( 
                in_channels=4, 
                out_channels=4, 
                kernel_size=(3,3,3),
                padding='same',
                dtype=torch.double
            ),
            nn.BatchNorm3d(4, dtype=torch.double),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(0.3),
            
            nn.Conv3d( 
                in_channels=4, 
                out_channels=8, 
                kernel_size=(1,3,3),
                padding='same',
                dtype=torch.double
            ),
            nn.BatchNorm3d(8, dtype=torch.double),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(0.3),
            
            nn.Conv3d( 
                in_channels=8, 
                out_channels=8, 
                kernel_size=(3,3,3),
                padding='same',
                dtype=torch.double
            ),
            nn.BatchNorm3d(8, dtype=torch.double),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(0.3),
            
            nn.MaxPool3d(
                kernel_size= (1, 2, 2)
            ),
            
            #post max-pooling
            nn.Conv3d( 
                in_channels=8, 
                out_channels=16, 
                kernel_size=(1,3,3),
                padding='same',
                dtype=torch.double
            ),
            nn.BatchNorm3d(16, dtype=torch.double),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(0.3),
            
            nn.Conv3d( 
                in_channels=16, 
                out_channels=16, 
                kernel_size=(3,3,3),
                padding='same',
                dtype=torch.double
            ),
            nn.BatchNorm3d(16, dtype=torch.double),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(0.3),
            
            nn.Conv3d( 
                in_channels=16, 
                out_channels=32, 
                kernel_size=(1,3,3),
                padding='same',
                dtype=torch.double
            ),
            nn.BatchNorm3d(32, dtype=torch.double),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(0.3),
            
            nn.Conv3d( 
                in_channels=32, 
                out_channels=32, 
                kernel_size=(3,3,3),
                padding='same',
                dtype=torch.double
            ),
            nn.BatchNorm3d(32, dtype=torch.double),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(0.3),
            
            nn.Conv3d( 
                in_channels=32, 
                out_channels=32, 
                kernel_size=(3,3,3),
                padding='same',
                dtype=torch.double
            ),
            nn.BatchNorm3d(32, dtype=torch.double),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(0.3)
            # nn.Linear(
            #     in_features = 32*self._target_shape_[0]*self._target_shape_[1]*self._target_shape_[2]//4, 
            #     out_features=self._sequence_embedding_features_
            # ),
            # nn.Dropout(p=dropout_rate),
            # nn.LeakyReLU(0.3),
        )
        
        return sequential_layers
    
    def __fusion_block__(self, adc_embedding, bval_embedding, ktrans_embedding, zone_embedding):
        fusion_embedding = torch.cat([adc_embedding, bval_embedding, ktrans_embedding, zone_embedding], dim=1)
        return fusion_embedding
        
    def __fc_block__(self, mode = 'classifier', dropout_rate = 0.25):
        if mode == 'classifier':
            sequential_layers = nn.Sequential(
                nn.Linear(
                    in_features = 3*self._sequence_embedding_features_, 
                    out_features=3*self._sequence_embedding_features_
                ),
                nn.BatchNorm1d(3*self._sequence_embedding_features_),
                nn.Dropout(p=dropout_rate),
                nn.LeakyReLU(0.3),
                nn.Linear(
                    in_features = 3*self._sequence_embedding_features_, 
                    out_features= 97
                ),
                nn.BatchNorm1d(97),
                nn.Dropout(p=dropout_rate),
                nn.LeakyReLU(0.3),
                 nn.Linear(
                    in_features = 97, 
                    out_features=1
                ),
                nn.Sigmoid()
            ) 
            return sequential_layers
        elif mode == 'contrastive':
            if self._sequence_embedding_features_  <= 256:
                sequential_layers = nn.Sequential(
                         nn.Linear(
                            in_features = self._sequence_embedding_features_ , 
                            out_features=self._sequence_embedding_features_
                        ),
                        #nn.BatchNorm1d(3*self._sequence_embedding_features_ + 3),
                        #nn.Dropout(p=dropout_rate),
                        nn.LeakyReLU(0.3),
                        nn.Linear(
                            in_features = self._sequence_embedding_features_ , 
                            out_features=128
                        )
                       
                ) 
                return sequential_layers
            else:
                sequential_layers = nn.Sequential(
                         nn.Linear(
                            in_features = self._sequence_embedding_features_, 
                            out_features=self._sequence_embedding_features_
                        ),
                        #nn.BatchNorm1d(3*self._sequence_embedding_features_ + 3),
                        #nn.Dropout(p=dropout_rate),
                        nn.LeakyReLU(0.3),
                        nn.Linear(
                            in_features = self._sequence_embedding_features_, 
                            out_features= 256
                        ),
                        #nn.BatchNorm1d(256),
                        #nn.Dropout(p=dropout_rate),
                        nn.LeakyReLU(0.3),
                        nn.Linear(
                            in_features = 256, 
                            out_features=128
                        )
                        
                ) 
                return sequential_layers
    
    def forward(self, x):
        t2 = x[0].double()
        adc = x[1].double()
        bval = x[2].double()
        
        t2 = t2.unsqueeze(1)
        print(f'Input shape t2: {t2.shape}')
        adc = adc.unsqueeze(1)
        bval = bval.unsqueeze(1)

        
        t2_embedding = self.t2_conv_branch(t2)
        print(f'shape t2_embedding apply backbone: {t2_embedding.shape}')
        
        adc_embedding = self.adc_conv_branch(adc)
        bval_embedding = self.bval_conv_branch(bval)
        fusion_embedding = torch.cat([t2_embedding, adc_embedding, bval_embedding], dim=1)
        print(f'Fusion shape: {fusion_embedding.shape}')
        fusion_embedding = fusion_embedding.view(fusion_embedding.shape[0],fusion_embedding.shape[1],-1)

        logits = self.fc_block(fusion_embedding)
        print(f'logits shape: {logits.shape}')
        return logits

class Contrastive_CB3(nn.Module): #Contrastivo sin SPD con los CB de JO
    def __init__(self, device, target_shape, sequence_embedding_features=64, mode='classifier'):
        super(Contrastive_CB3, self).__init__()
        self._target_shape_ = target_shape
        self._sequence_embedding_features_ = sequence_embedding_features
        self._mode_ = mode
        self.t2_conv_branch =  CB3()
        self.adc_conv_branch =  CB3()
        self.bval_conv_branch =  CB3()
        self.fc_block = self.__fc_block__(mode)
  
        
    def __fc_block__(self, mode = 'classifier', dropout_rate = 0.25):
        if mode == 'classifier':
            sequential_layers = nn.Sequential(
                nn.Linear(
                    in_features = 3*self._sequence_embedding_features_, 
                    out_features=3*self._sequence_embedding_features_
                ),
                nn.BatchNorm1d(3*self._sequence_embedding_features_),
                nn.Dropout(p=dropout_rate),
                nn.LeakyReLU(0.3),
                nn.Linear(
                    in_features = 3*self._sequence_embedding_features_, 
                    out_features= 97
                ),
                nn.BatchNorm1d(97),
                nn.Dropout(p=dropout_rate),
                nn.LeakyReLU(0.3),
                 nn.Linear(
                    in_features = 97, 
                    out_features=1
                ),
                nn.Sigmoid()
            ) 
            return sequential_layers
        elif mode == 'contrastive':
            
            
            if self._sequence_embedding_features_  <= 256:
                sequential_layers = nn.Sequential(
                         nn.Linear(
                            in_features = self._sequence_embedding_features_ ,   #192, 96 * 96, 192
                            out_features = self._sequence_embedding_features_
                        ),
                        #nn.BatchNorm1d(3*self._sequence_embedding_features_ + 3),
                        #nn.Dropout(p=dropout_rate),
                        nn.LeakyReLU(0.3),
                        nn.Linear(
                            in_features = self._sequence_embedding_features_ , 
                            out_features=128
                        )
                       
                ) 
                return sequential_layers
            else:
                sequential_layers = nn.Sequential(
                         nn.Linear(
                            in_features = self._sequence_embedding_features_, 
                            out_features=1024
                        ),
                        #nn.BatchNorm1d(3*self._sequence_embedding_features_ + 3),
                        #nn.Dropout(p=dropout_rate),
                        nn.LeakyReLU(0.3),
                        nn.Linear(
                            in_features = 1024, 
                            out_features= 256
                        ),
                        #nn.BatchNorm1d(256),
                        #nn.Dropout(p=dropout_rate),
                        nn.LeakyReLU(0.3),
                        nn.Linear(
                            in_features = 256, 
                            out_features=128
                        )
                        
                ) 
                return sequential_layers
    
    def forward(self, x):
        t2 = x[0].double()
        adc = x[1].double()
        bval = x[2].double()
        
        t2 = t2.unsqueeze(1)
        #print(f'Input shape t2: {t2.shape}') #([32, 1, 12, 32, 32])
        adc = adc.unsqueeze(1)
        bval = bval.unsqueeze(1)

        
        t2_embedding = self.t2_conv_branch(t2)
        #print(f'shape t2_embedding apply backbone: {t2_embedding.shape}') #([32, 64, 6, 4, 4])
        
        adc_embedding = self.adc_conv_branch(adc)
        bval_embedding = self.bval_conv_branch(bval)
        fusion_embedding = torch.cat([t2_embedding, adc_embedding, bval_embedding], dim=1)
        #print(f'Fusion shape: {fusion_embedding.shape}') #([32, 192, 6, 4, 4])
        fusion_embedding = fusion_embedding.view(fusion_embedding.shape[0], -1).float() #Se lo quité para entrenar.float() poner para emmbedings
        #print(f'fusion_embedding despues de concatenar: {fusion_embedding.shape} DTYPE:{fusion_embedding.dtype}') #([32, 192*6*4*4])
        
        
        logits = self.fc_block(fusion_embedding)
        #print(f'logits shape: {logits.shape}')
        return logits
    
class SPDnet(nn.Module):
    def __init__(self, device):
        super(__class__, self).__init__()
        dim_input = 96 #dimension entrada (96X96)                
        n_classes = 2
        
        self.bimap1 = nn_spd.BiMap(1, 1, dim_input, dim_input // 2, dtype = torch.double, device = device) #BIMAP layer -> Channels out, channels in, dim_Spd_in, dim_spd_out
        self.reig1 = nn_spd.ReEig()  
        
        self.bimap2 = nn_spd.BiMap(1, 1, dim_input //2, dim_input //4, dtype = torch.double, device = device) #BIMAP layer -> Channels out, channels in, dim_Spd_in, dim_spd_out
        self.reig2 = nn_spd.ReEig()
                     
        self.logeig = nn_spd.LogEig()
        
        self.indices = torch.triu_indices(dim_input // 4, dim_input // 4)
        
        self.output_layers = nn.Sequential(
                    nn.Linear(in_features=300, out_features=64), #Tamaño de la Triu
                    nn.Linear(in_features=64, out_features=32),
                    nn.Linear(in_features=32, out_features=2),
        )
        #print(self.linear.shape)
    def forward(self, x):
        
        x = self.reig1(self.bimap1(x))
        x = self.reig2(self.bimap2(x))
        x = self.logeig(x).squeeze(1)
        
        # print(f'x shape: {x.shape}')
        
        upper_triangular = x[:, self.indices[0], self.indices[1]].float()
        # print(f"upper_triangular shape: {upper_triangular.shape}")
        
        out = self.output_layers(upper_triangular)
        
        return out
    
class SPDnet1Bire(nn.Module):
    def __init__(self, device):
        super(__class__, self).__init__()
        dim_input = 96 #dimension entrada (96X96)                
        n_classes = 2
        
        self.bimap1 = nn_spd.BiMap(1, 1, dim_input, dim_input // 2, dtype = torch.double, device = device) #BIMAP layer -> Channels out, channels in, dim_Spd_in, dim_spd_out
        self.reig1 = nn_spd.ReEig()  
         
        self.logeig = nn_spd.LogEig()
        
        self.indices = torch.triu_indices(dim_input // 2, dim_input // 2)
        
        self.output_layers = nn.Sequential(
                    nn.Linear(
                        in_features=1176, #Tamaño de la Triu
                        out_features=256
                        ), 
                    nn.BatchNorm1d(256),
                    nn.Dropout(p=0.25),
                    nn.LeakyReLU(0.3),
                    nn.Linear(
                        in_features=256,
                        out_features=128
                        ),
                    nn.BatchNorm1d(128),
                    nn.Dropout(p=0.25),
                    nn.LeakyReLU(0.3),
                    nn.Linear(
                        in_features=128,
                        out_features=64
                        ),
                    nn.BatchNorm1d(64),
                    nn.Dropout(p=0.25),
                    nn.LeakyReLU(0.3),
                    nn.Linear(
                        in_features=64,
                        out_features=32
                        ),
                    nn.BatchNorm1d(32),
                    nn.Dropout(p=0.25),
                    nn.LeakyReLU(0.3),
                    nn.Linear(
                        in_features=32,
                        out_features=n_classes
                        ),
        )
        #print(self.linear.shape)
    def forward(self, x):
        
        x = self.reig1(self.bimap1(x))
        x = self.logeig(x).squeeze(1)
        
        # print(f'x shape: {x.shape}')
        
        upper_triangular = x[:, self.indices[0], self.indices[1]].float()
        # print(f"upper_triangular shape: {upper_triangular.shape}")
        
        out = self.output_layers(upper_triangular)
        
        return out

# class SPDnet1Bire(nn.Module):
#     def __init__(self, device):
#         super(__class__, self).__init__()
#         dim_input = 96 #dimension entrada (96X96)                
#         n_classes = 2
        
#         self.bimap1 = nn_spd.BiMap(1, 1, dim_input, dim_input // 2, dtype = torch.double, device = device) #BIMAP layer -> Channels out, channels in, dim_Spd_in, dim_spd_out
#         self.reig1 = nn_spd.ReEig()  
         
#         self.logeig = nn_spd.LogEig()
        
#         self.indices = torch.triu_indices(dim_input // 2, dim_input // 2)
        
#         self.output_layers = nn.Sequential(
#                     nn.Linear(in_features=1176, out_features=64), #Tamaño de la Triu
#                     nn.Linear(in_features=64, out_features=32),
#                     nn.Linear(in_features=32, out_features=2),
#         )
#         #print(self.linear.shape)
#     def forward(self, x):
        
#         x = self.reig1(self.bimap1(x))
#         x = self.logeig(x).squeeze(1)
        
#         # print(f'x shape: {x.shape}')
        
#         upper_triangular = x[:, self.indices[0], self.indices[1]].float()
#         # print(f"upper_triangular shape: {upper_triangular.shape}")
        
#         out = self.output_layers(upper_triangular)
        
#         return out


class Contrastive_CB3_Combined(nn.Module): #Contrastivo sin SPD con los CB de JO
    def __init__(self, device, target_shape, sequence_embedding_features=64, mode='classifier'):
        super(Contrastive_CB3_Combined, self).__init__()
        self._target_shape_ = target_shape
        self._sequence_embedding_features_ = sequence_embedding_features
        self._mode_ = mode
        self.t2_conv_branch =  CB3()
        self.adc_conv_branch =  CB3()
        self.bval_conv_branch =  CB3()
        self.fc_block = self.__fc_block__(mode)
        
        #SPD NETWORK
        self.covariance  = nn_spd.CovPool()
        self.spd_module = nn.Sequential(
            nn_spd.BiMap(1 , 1, 3*64, 64, dtype = torch.double, device = device), #(No.Mat output,No.matrices input,Size input,size output)
            nn_spd.ReEig(),
            nn_spd.LogEig()
        )
        self.indices = torch.triu_indices(64,64)
        
        #Densa
        self.output_layers = nn.Sequential(
            nn.Linear(in_features=2080, out_features=512),
            nn.Linear(in_features=512, out_features=128),
            nn.Linear(in_features=128, out_features=64),
            nn.Linear(in_features=64, out_features=2),
        )
            
  
        
    def __fc_block__(self, mode = 'classifier', dropout_rate = 0.25):
        if mode == 'classifier':
            sequential_layers = nn.Sequential(
                nn.Linear(
                    in_features = 3*self._sequence_embedding_features_, 
                    out_features=3*self._sequence_embedding_features_
                ),
                nn.BatchNorm1d(3*self._sequence_embedding_features_),
                nn.Dropout(p=dropout_rate),
                nn.LeakyReLU(0.3),
                nn.Linear(
                    in_features = 3*self._sequence_embedding_features_, 
                    out_features= 97
                ),
                nn.BatchNorm1d(97),
                nn.Dropout(p=dropout_rate),
                nn.LeakyReLU(0.3),
                 nn.Linear(
                    in_features = 97, 
                    out_features=1
                ),
                nn.Sigmoid()
            ) 
            return sequential_layers
        elif mode == 'contrastive':
            
            
            if self._sequence_embedding_features_  <= 256:
                sequential_layers = nn.Sequential(
                         nn.Linear(
                            in_features = self._sequence_embedding_features_ ,   #192, 96 * 96, 192
                            out_features = self._sequence_embedding_features_
                        ),
                        #nn.BatchNorm1d(3*self._sequence_embedding_features_ + 3),
                        #nn.Dropout(p=dropout_rate),
                        nn.LeakyReLU(0.3),
                        nn.Linear(
                            in_features = self._sequence_embedding_features_ , 
                            out_features=128
                        )
                       
                ) 
                return sequential_layers
            else:
                sequential_layers = nn.Sequential(
                         nn.Linear(
                            in_features = self._sequence_embedding_features_, 
                            out_features=1024
                        ),
                        #nn.BatchNorm1d(3*self._sequence_embedding_features_ + 3),
                        #nn.Dropout(p=dropout_rate),
                        nn.LeakyReLU(0.3),
                        nn.Linear(
                            in_features = 1024, 
                            out_features= 256
                        ),
                        #nn.BatchNorm1d(256),
                        #nn.Dropout(p=dropout_rate),
                        nn.LeakyReLU(0.3),
                        nn.Linear(
                            in_features = 256, 
                            out_features=128
                        )
                        
                ) 
                return sequential_layers
    
    def forward(self, x):
        t2 = x[0].double()
        adc = x[1].double()
        bval = x[2].double()
        
        t2 = t2.unsqueeze(1)
        #print(f'Input shape t2: {t2.shape}') #([32, 1, 12, 32, 32])
        adc = adc.unsqueeze(1)
        bval = bval.unsqueeze(1)

        
        t2_embedding = self.t2_conv_branch(t2)
        # print(f'shape t2_embedding apply backbone: {t2_embedding.shape}') #([32, 64, 6, 4, 4])
        
        adc_embedding = self.adc_conv_branch(adc)
        bval_embedding = self.bval_conv_branch(bval)
        fusion_embedding = torch.cat([t2_embedding, adc_embedding, bval_embedding], dim=1)
        #print(f'Fusion shape: {fusion_embedding.shape}') #([32, 192, 6, 4, 4])
        fusion_embedding_CL = fusion_embedding.view(fusion_embedding.shape[0], -1).float() #Se lo quité para entrenar.float() poner para emmbedings
        #print(f'fusion_embedding despues de concatenar: {fusion_embedding.shape} DTYPE:{fusion_embedding.dtype}') #([32, 192*6*4*4])
        
        
        logits = self.fc_block(fusion_embedding_CL) #Shape [32, 128]
        #print(f'logits shape: {logits.shape}')
        
        
        #SPD NETWORK
        fusion_embedding_SPD = fusion_embedding.view(fusion_embedding.shape[0],fusion_embedding.shape[1],-1)#.float() #Shape (32, 192, 6*4*4)
        covariance = self.covariance(fusion_embedding_SPD) # hace w * w.T Shape [32, 1, 192, 192]
        
        spd_output = self.spd_module(covariance).squeeze(1) # Shape [32, 1, 64, 64]
        
        upper_triangular = spd_output[:, self.indices[0], self.indices[1]] #Shape = [32, 2080]
        upper_triangular = upper_triangular.to(torch.float32) #para validación
        
        out = self.output_layers(upper_triangular) #Shape [32, 2]
        # print(out.shape)
        
        
        
        
        
        
        return logits, out # embeddings, 2 classes


    
    
    
    
