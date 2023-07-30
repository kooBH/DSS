#!/bin/bash

TASK=UDSS

VERSION=v58
python src/evaluation.py --config ./config/${TASK}/${VERSION}.yaml --default ./config/${TASK}/default.yaml  --chkpt /home/nas/user/kbh/${TASK}/chkpt/${VERSION}/bestmodel.pt --device cuda:0 --dir_out /home/data2/kbh/DSS/eval/${VERSION}
