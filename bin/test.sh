#!/usr/bin/env bash

#rm -rf ./ckpt
#rm -rf ./export_dir
sh stop.sh

sleep 1
export TF_CONFIG='{"cluster": {"worker": ["localhost:2222"], "ps": ["localhost:2223"], "chief": ["localhost:2224"]}, "task": {"type": "ps", "index": 0}}'
python test.py --mode train --task_type "ps" --task_idx "0" &

sleep 1
export TF_CONFIG='{"cluster": {"worker": ["localhost:2222"], "ps": ["localhost:2223"], "chief": ["localhost:2224"]}, "task": {"type": "worker", "index": 0}}'
python test.py --mode train --task_type "worker" --task_idx "0" &

sleep 1
export TF_CONFIG='{"cluster": {"worker": ["localhost:2222"], "ps": ["localhost:2223"], "chief": ["localhost:2224"]}, "task": {"type": "chief", "index": 0}}'
python test.py --mode train --task_type "chief" --task_idx "0" &


echo "ok>>>"
