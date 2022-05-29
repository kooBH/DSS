#!/bin/bash  

VERSION=v5

ROOT_INPUT=/home/data2/kbh/DESED/soundbank/audio/train/soundbank/foreground/
ROOT_OUTPUT=/home/data2/kbh/LGE/DESED_train_${VERSION}
N_OUTPUT=100000
#python DESED.py -i ${ROOT_INPUT} -o ${ROOT_OUTPUT} -n ${N_OUTPUT}

ROOT_INPUT=/home/data2/kbh/DESED/soundbank/audio/validation/soundbank/foreground/
ROOT_OUTPUT=/home/data2/kbh/LGE/DESED_test_${VERSION}
N_OUTPUT=5000
python DESED.py -i ${ROOT_INPUT} -o ${ROOT_OUTPUT} -n ${N_OUTPUT}

ROOT_INPUT=/home/data2/kbh/DESED/soundbank/audio/validation/soundbank/foreground/
ROOT_OUTPUT=/home/data2/kbh/LGE/DESED_eval_simu_${VERSION}
N_OUTPUT=100
python DESED.py -i ${ROOT_INPUT} -o ${ROOT_OUTPUT} -n ${N_OUTPUT}
