#!/bin/bash

DATE=20220803
#DATA=DESED_5000_v2
#DATA=v9_oversample
DEVICE=cuda:1

#python src/inference.py -c config/${version}.yaml --chkpt /home/nas/user/kbh/DOA-Audio-Separation/chkpt/${version}/bestmodel.pt -i /home/data2/kbh/LGE/${DATA} -o /home/nas/user/kbh/DOA-Audio-Separation/infer_${version}_${DATE} -d ${DEVICE} 




for version in dv10b_v56 ; do
  python src/inference.py -c config/${version}.yaml --chkpt /home/nas/user/kbh/DOA-Audio-Separation/chkpt/${version}/bestmodel.pt -i /home/data2/kbh/LGE/${DATA} -o /home/data2/kbh/LGE/infer_${version}_${DATE}_${DATA} -d ${DEVICE} 
done
