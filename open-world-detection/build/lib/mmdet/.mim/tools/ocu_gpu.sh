#!/bin/bash
var=0
ocp_memory=${2:-500}
while [ $var -eq 0 ]
echo 'waiting for available gpu...'
do
    count=0
    for i in $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    do
        if [ $i -lt ocp_memory ]
        then
            echo 'GPU'$count' is avaiable'
            CUDA_VISIBLE_DEVICES=$count python ${1}
            var=1
            break
        fi
        count=$(($count+1))    
    done    
done

# sh ocu_gpu.sh test.py 500