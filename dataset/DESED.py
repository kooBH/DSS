import sys,os
sys.path.append("gpuRIR")

from mixing import generate

import glob
import argparse

import numpy as np
import librosa
import torch

# utils
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Due to 'PySoundFile failed. Trying audioread instead' 
import warnings
warnings.filterwarnings('ignore')

# param
parser = argparse.ArgumentParser()
parser.add_argument('--input_root', '-i', type=str, required=True)
parser.add_argument('--output_root', '-o', type=str, required=True)
parser.add_argument('--n_output', '-n', type=int, required=True)
args = parser.parse_args()

## ROOT
root = args.input_root
output_root = args.output_root
n_output = args.n_output

## PATH
target_list = [x for x in glob.glob(os.path.join(root,'**','*.wav'),recursive=True)]
#target_list += [x for x in glob.glob(os.path.join("/home/data/kbh/CHiME4/isolated_ext/","tr*","*.CH1.Clean.wav"))]

print("Target Files : {}".format(len(target_list)))

## Gen List, 
# currently, np.random seed is fixed in MP

list_sources = []
for i in range(n_output):
    n_source = np.random.randint(low=1,high=4+1)
    list_sources.append(np.random.choice(target_list,n_source))

def process(idx):
    # gen random parameters
    #n_source = np.random.randint(low=1,high=4)
    #list_sources = np.random.choice(target_list,n_source)
    #generate(output_root,list_sources[idx],idx,50,shift=128,match="1sec",fix=True,max_SIR=5)
    #generate(output_root,list_sources[idx],idx,50,shift=128,match="1sec",fix=True,max_SIR=5,max_RT60=0.15)

    ## 2022-05-09 v4 : original 'signals' scale
    #generate(output_root,list_sources[idx],idx,50,shift=128,match="1sec",fix=True,max_SIR=5,max_RT60=0.15,norm_signals=False)

    ## 2022-05-24 v5 : more hard
    #generate(output_root,list_sources[idx],idx,50,shift=128,match="1sec",fix=True,max_SIR=10,max_RT60=0.8,norm_signals=False)

    ## 2022-05-30 v6 : more speech , less SIR
    #generate(output_root,list_sources[idx],idx,50,shift=128,match="1sec",fix=False,max_SIR=5,max_RT60=0.8,norm_signals=False)

    ## 2022-06-08 v7 : too short ? 
    #generate(output_root,list_sources[idx],idx,50,shift=128,match="4sec",fix=True,max_SIR=10,max_RT60=0.8,norm_signals=False)

    ## 2022-07-03 v8 : Renewal 
    generate(output_root,list_sources[idx],idx,50,shift=128,match="4sec",fix=True,max_SIR=10,max_RT60=0.8,norm_signals=False)




if __name__=='__main__': 
    cpu_num = cpu_count()
    cpu_num = 16

    os.makedirs(os.path.join(output_root),exist_ok=True)

    arr = list(range(n_output))

    with Pool(cpu_num) as p:
        r = list(tqdm(p.imap(process, arr), total=len(arr),ascii=True,desc='processing'))
