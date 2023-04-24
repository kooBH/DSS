#!/bin/bash

DEVICE=$1
#DEVICE=drone
#DEVICE=preprocess

#DEVICE=AIO
#DEVICE=train


list="$(ps -aux)"

#echo "$list"

while read -r p; do
  if grep -q "${DEVICE}" <<< $p; then
    a=${p[0]}
    idx=0
    for i in $a; do
      if [ $idx -eq 1 ]
        then
           echo ${p}
           sudo kill -9 $i
           #sudo kill -19 $i
           echo [kill] ${p}
      fi
      idx=$(($idx+1))
    done
  fi
done <<<"$list"

#echo =======
