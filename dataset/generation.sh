#!/bin/bash  

ROOT_INPUT=/home/data2/kbh/DESED/soundbank/audio/train/soundbank/foreground/
ROOT_OUTPUT=/home/data2/kbh/LGE/v2
N_OUTPUT=50

python DESED.py -i ${ROOT_INPUT} -o ${ROOT_OUTPUT} -n ${N_OUTPUT}
