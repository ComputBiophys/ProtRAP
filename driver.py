import numpy as np
import argparse,os
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
    argparser.add_argument('--ten_average',type=bool ,default=False)
    args=argparser.parse_args()
    seq=load_fasta(args.data+'.fasta')
    feat=load_feat(args.data+'.feat')
    seqv=seq2arr(seq)
    value=np.concatenate((seqv,feat[:,:20],feat[:,-3:]),axis=1)

    model_path=lambda x:'models/model_'+str(x)+'.pjit'
    models=[]
    for i in range(10):
        if not os.path.exists(model_path(i)):
            print('Downloading model_'+str(i))
            url='https://github.com/ComputBiophys/ProtRA/releases/download/weights/model_'+str(i)+'.pjit'
            import requests
            r  = requests.get(url,stream=True)
            with open (model_path(i),'wb') as f:
                f.write(r.content)
        model=torch.jit.load(os.path.join(os.getcwd(),model_path(i))).to(args.device)
        models.append(model)
        if not args.ten_average:
            break
    model.eval()
    with torch.no_grad():
        value=torch.tensor(value, dtype=torch.float32).unsqueeze(0)
        results=[]
        for model in models:
            result=model(value)
            result=result.numpy()[0,...]
            results.append(result)
    results=np.array(results)
    results=np.mean(results,axis=0)
    np.savetxt(args.data+'.csv',results,delimiter=',')
