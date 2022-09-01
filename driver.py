import numpy as np
import argparse,os
import pandas as pd
from models import *
github_url='https://github.com/ComputBiophys/ProtRAP/releases/download/weights/model'
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
def load_data(path):
    seq=load_fasta(path+'.fasta')
    feat=load_feat(path+'.feat')
    seqv=seq2arr(seq)
    value=np.concatenate((seqv,feat[:,:20],feat[:,-3:]),axis=1)
    return value

def calculate(models,value):
    with torch.no_grad():
        value=torch.tensor(value, dtype=torch.float32).unsqueeze(0)
        results=[]
        for model in models:
            result=model(value)
            result=result.numpy()[0,...]
            results.append(result)
    results=np.array(results)
    results=np.mean(results,axis=0)
    return results

if __name__=='__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--device', default='cpu', help='pytorch device name, default cpu')
    argparser.add_argument('--input_path', default='prot', help='input data path')
    argparser.add_argument('--input_type', default='file', help='file or dir')
    argparser.add_argument('--ten_average',type=bool ,default=False)
    args=argparser.parse_args()

    model_path=lambda x:'models/model_'+str(x)+'.pjit'
    models=[]
    for i in range(10):
        if not os.path.exists(model_path(i)):
            print('Downloading model_'+str(i))
            url=github_url+str(i)+'.pjit'
            import requests
            r  = requests.get(url,stream=True)
            with open (model_path(i),'wb') as f:
                f.write(r.content)
        model=torch.jit.load(os.path.join(os.getcwd(),model_path(i))).to(args.device).eval()
        models.append(model)
        if not args.ten_average:
            break

    if args.input_type=='file':
        value=load_data(args.input_path)
        results=calculate(models, value)
        np.savetxt(args.input_path+'.csv',results,delimiter=',')
        pass
    elif args.input_type=='dir':
        files=os.listdir(args.input_path)
        feats=set(filter(lambda x:x.split('.')[-1]=='feat',files))
        feats=set(map(lambda x:x.split('.')[0],feats))
        fastas=set(filter(lambda x:x.split('.')[-1]=='fasta',files))
        fastas=set(map(lambda x:x.split('.')[0],fastas))
        files=feats&fastas
        for fname in files:
            input_path=os.path.join(args.input_path,fname)
            value=load_data(input_path)
            results=calculate(models, value)
            np.savetxt(input_path+'.csv',results,delimiter=',')
    else:
        print('invalid input type')
