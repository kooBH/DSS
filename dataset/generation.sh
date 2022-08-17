#!/bin/bash  

VERSION=v7


ROOT_OUTPUT=/home/data2/kbh/LGE/DESED_test_${VERSION}
N_OUTPUT=5000
python DESED.py -i ${ROOT_INPUT} -o ${ROOT_OUTPUT} -n ${N_OUTPUT}

ROOT_OUTPUT=/home/data2/kbh/LGE/DESED_eval_simu_${VERSION}
N_OUTPUT=100
python DESED.py -i ${ROOT_INPUT} -o ${ROOT_OUTPUT} -n ${N_OUTPUT}

ROOT_INPUT=/home/data2/kbh/DESED/soundbank/audio/train/soundbank/foreground/
ROOT_OUTPUT=/home/data2/kbh/LGE/DESED_train_${VERSION}
N_OUTPUT=50000
python DESED.py -i ${ROOT_INPUT} -o ${ROOT_OUTPUT} -n ${N_OUTPUT}


## TEMP

#ROOT_INPUT=/home/data2/kbh/DESED/soundbank/audio/validation/soundbank/foreground/Speech/
#ROOT_OUTPUT=/home/data2/kbh/LGE/20220601_exp
#N_OUTPUT=100
#python DESED.py -i ${ROOT_INPUT} -o ${ROOT_OUTPUT} -n ${N_OUTPUT}

