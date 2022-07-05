import numpy as np
import argparse
import pandas as pd
from models import *
def load_fasta(path):
    with open(path,'r') as f:
        lines=f.readlines()
    seq=''
    for line in lines[1:]:
        seq+=line
    return seq.replace('\n','')
def seq2arr(seq):
    seq=np.array(list(seq))
    arr=np.zeros((len(seq),20))
    data=list('ACDEFGHIKLMNPQRSTVWY')
    for i in range(20):
        arr[seq==data[i],i]=1
    return arr
def load_feat(path):
    val=pd.read_table(path,header=None,sep=' ').values[:,:-1]
    return val
if __name__=='__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device', default='cpu', help='pytorch device name, default cpu')
    argparser.add_argument('--data', default='prot', help='data path')
    argparser.add_argument('--model_path',  help='model path')
    args=argparser.parse_args()
    seq=load_fasta(args.data+'.fasta')
    feat=load_feat(args.data+'.feat')
    seqv=seq2arr(seq)
    value=np.concatenate((seqv,feat[:,:20],feat[:,-3:]),axis=1)

    model=torch.load(args.model_path,map_location=args.device) # or cuda
    model.eval()
    with torch.no_grad():
        value=torch.tensor(value, dtype=torch.float32).unsqueeze(0)
        result=model(value)
        result=result.numpy()[0,...]