#!/bin/bash

EST=/home/data2/kbh/LGE/infer_dv5_v13_20220530_MLDR_alt/
REF=/home/data2/kbh/LGE/DESED_eval_simu_v5/

python src/score_beam.py -i ${EST} -r ${REF} -n 100

