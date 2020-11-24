#!/bin/sh
readonly LOG_FILE="service.log"
touch $LOG_FILE
export http_proxy="http://localhost:8123"
export https_proxy="http://localhost:8123"
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate myenv3 
exec 1>$LOG_FILE
exec 2>&1
python word2vec-api.py --w2v_model ~/data/SO_vectors_200.bin --binary true
