#!/bin/bash

DEVICE=cuda:1

#DATA=AIO_eval_simu_v8
#DATA=DESED_eval_simu_v7
DATA=DESED_eval_simu_v5
#DATA=DESED_5000_v2
#DATA=sample_v2

version=dv5_v24
python src/score_sep.py -c config/${version}.yaml --chkpt /home/nas/user/kbh/DOA-Audio-Separation/chkpt/${version}/bestmodel.pt -i /home/data2/kbh/LGE/${DATA} -d ${DEVICE}

#version=dv7_v33
#python src/score_sep.py -c config/${version}.yaml --chkpt /home/nas/user/kbh/DOA-Audio-Separation/chkpt/${version}/bestmodel.pt -i /home/data2/kbh/LGE/${DATA} -d ${DEVICE}
#version=dv8_v33
#python src/score_sep.py -c config/${version}.yaml --chkpt /home/nas/user/kbh/DOA-Audio-Separation/chkpt/${version}/bestmodel.pt -i /home/data2/kbh/LGE/${DATA} -d ${DEVICE}
#version=dv8_v37
#python src/score_sep.py -c config/${version}.yaml --chkpt /home/nas/user/kbh/DOA-Audio-Separation/chkpt/${version}/bestmodel.pt -i /home/data2/kbh/LGE/${DATA} -d ${DEVICE}
