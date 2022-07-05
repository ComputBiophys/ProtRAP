import math

import torch
import torch.nn as nn
import torch.nn.functional as nnF
keep_prob=0.9

# Customized convolution module, integrately perform transpose and multi-size convolution
class TransConv(nn.Module):
    def __init__(self,input_dim=46,output_dim=48):#,size=[3,7,11],padding=[1,3,5]):
        super().__init__()

        self.cnn1=nn.Conv1d(input_dim,output_dim,3,padding=1)
        self.cnn2=nn.Conv1d(input_dim,output_dim,7,padding=3)
        self.cnn3=nn.Conv1d(input_dim,output_dim,11,padding=5)
    def forward(self,x):
        x=x.permute(0,2,1)

        x1=self.cnn1(x)
        x2=self.cnn2(x)
        x3=self.cnn3(x)
        x=torch.cat((x1,x2,x3), -2)
        x=nnF.relu(x)
        x=x.permute(0,2,1)
        return x

def get_length(x):
    xsum=(x>0).sum(axis=-1)
    length=(xsum>0).sum(axis=-1)
    return length

def len_1k_pad(x):
    length=x.shape[1]
    if 1000>length:
        return nnF.pad(x,(0,0,0,1000-length,0,0))
    else:
        return x

def length_pad(x):
    length=x.shape[1]
    newLength=4*math.ceil(length/4)
    if newLength>length:
        return nnF.pad(x,(0,0,0,newLength-length,0,0))
    else:
        return x

class pack_GRU(nn.Module):
    def __init__(self,input_dim,n_hidden=128,stack=3):
        super().__init__()
        self.biRNN=nn.GRU(input_dim,n_hidden,stack,batch_first=True,dropout=1-keep_prob,bidirectional=True)
    def forward(self,x,length):
        pack_x=nn.utils.rnn.pack_padded_sequence(x,length.to('cpu'),batch_first=True,enforce_sorted=False)
        x,_=self.biRNN(pack_x)
        padx,lengthx=nn.utils.rnn.pad_packed_sequence(x,batch_first=True)
        return padx

class Basic_Model(nn.Module):
    def __init__(self,input_dim,n_hidden=128):
        super().__init__()
        self.trans_conv=TransConv(input_dim, 48)
        self.biRNN=pack_GRU(48*3,n_hidden)
        self.final_linear_stack=nn.Sequential(
            nn.Linear(n_hidden*2,128),nn.SELU(),
            nn.Linear(128,32),nn.SELU(),
            nn.Linear(32,3),nn.Softmax(dim=-1)
        )
    def forward(self,x):
        lengths=get_length(x)
        x=self.trans_conv(x)
        x=self.biRNN(x,lengths)
        x=self.final_linear_stack(x)
        return x 

models_name_dict={}
class Transformer_light_Model(nn.Module):
    def __init__(self,input_dim=43,n_hidden=128):
        super().__init__()
        attention_dim=64
        self.trans_conv=TransConv(input_dim, 48)
        self.drop=nn.Dropout(0.2)
        self.biRNN=pack_GRU(48*3,n_hidden,2)
        encoder_layer=nn.TransformerEncoderLayer(d_model=attention_dim, nhead=4,batch_first=True)
        self.encoder= nn.TransformerEncoder(encoder_layer,num_layers=2)
        self.biRNN2=pack_GRU(n_hidden*2+attention_dim,n_hidden,1)
        self.linear1=nn.Linear(n_hidden*2,attention_dim)
        self.fc1=nn.Sequential(nn.Linear(n_hidden*2+attention_dim,196),nn.SELU(),nn.Dropout(0.3))
        self.final_linear_stack=nn.Sequential(
            nn.Linear(196,64),nn.SELU(),nn.Dropout(0.1),
            nn.Linear(64,3),nn.Softmax(dim=-1)
        )
    def forward(self,x):
        lengths=get_length(x)
        x=self.trans_conv(x)
        x=self.drop(x)
        bix=self.biRNN(x,lengths)
        att_x=self.linear1(bix)
        att=self.encoder(att_x)
        x=torch.cat((bix,att),-1)
        bix=self.biRNN2(x,lengths)
        x=torch.cat((bix,att),-1)
        x=self.fc1(x)
        x=self.final_linear_stack(x)
        return x
models_name_dict['Transformer_light']=Transformer_light_Model

class Pad_Model(nn.Module):
    def __init__(self,input_dim=43,n_hidden=128):
        super().__init__()
        self.trans_conv=TransConv(input_dim, 48)
        self.biRNN=pack_GRU(48*3,128)
        self.biRNN2=pack_GRU(256,128,1)
        self.final_linear_stack=nn.Sequential(
            nn.Linear(n_hidden*2,144),nn.SELU(),
            nn.Linear(144,32),nn.SELU(),
            nn.Linear(32,3),nn.Softmax(dim=-1)
        )

    def forward(self,x):
        lengths=get_length(x)
        x=self.trans_conv(x)
        x=self.biRNN(x,lengths)
        x2=self.biRNN2(x,lengths)
        x=self.final_linear_stack(x+x2)
        return x 
models_name_dict['Pad']=Pad_Model

class Pad3_Model(nn.Module):
    def __init__(self,input_dim=43,n_hidden=128):
        super().__init__()
        self.trans_conv=TransConv(input_dim, 48)
        self.biRNN=pack_GRU(48*3,128)
        self.biRNN2=pack_GRU(256,128,3)
        self.final_linear_stack=nn.Sequential(
            nn.Linear(n_hidden*2,144),nn.SELU(),
            nn.Linear(144,32),nn.SELU(),
            nn.Linear(32,3),nn.Softmax(dim=-1)
        )

    def forward(self,x):
        lengths=get_length(x)
        x=self.trans_conv(x)
        x=self.biRNN(x,lengths)
        x2=self.biRNN2(x,lengths)
        x=self.final_linear_stack(x+x2)
        return x 
models_name_dict['Pad3']=Pad3_Model

class Pad5_Model(nn.Module):
    def __init__(self,input_dim=43,n_hidden=128):
        super().__init__()
        self.trans_conv=TransConv(input_dim, 48)
        self.biRNN=pack_GRU(48*3,128)
        self.biRNN2=pack_GRU(256,128,5)
        self.final_linear_stack=nn.Sequential(
            nn.Linear(n_hidden*2,144),nn.SELU(),
            nn.Linear(144,32),nn.SELU(),
            nn.Linear(32,3),nn.Softmax(dim=-1)
        )

    def forward(self,x):
        lengths=get_length(x)
        x=self.trans_conv(x)
        x=self.biRNN(x,lengths)
        x2=self.biRNN2(x,lengths)
        x=self.final_linear_stack(x+x2)
        return x 
models_name_dict['Pad5']=Pad5_Model
