#!/bin/bash  

ROOT_INPUT=/home/data2/kbh/DESED/soundbank/audio/train/soundbank/foreground/
ROOT_OUTPUT=/home/data2/kbh/LGE/DESED_100000_v4
N_OUTPUT=100000
python DESED.py -i ${ROOT_INPUT} -o ${ROOT_OUTPUT} -n ${N_OUTPUT}

ROOT_INPUT=/home/data2/kbh/DESED/soundbank/audio/validation/soundbank/foreground/
ROOT_OUTPUT=/home/data2/kbh/LGE/DESED_5000_v4
N_OUTPUT=5000
python DESED.py -i ${ROOT_INPUT} -o ${ROOT_OUTPUT} -n ${N_OUTPUT}
