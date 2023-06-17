#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}

if [ "$1" = "" ]; then
  time_str=ckpt
else
  time_str=$1
fi
# cp -r ${model_dir}/ckpt ${model_dir}/ckpt_tmp

# sh ${code_dir}/bin/stop.sh
# python3 ${main_py} --model_dir "${model_dir}/ckpt_tmp/join" --mode train --type "join" --warm_path "${model_dir}/ckpt_tmp/update" \
#   --data_path "${nas_path}" --time_str "${time_str}" --time_format "%Y%m%d/*/*/*.gz" --lr 0.0

sh ${code_dir}/bin/stop.sh
TF_CONFIG='{}'
python3 ${code_dir}/main.py --model_dir ${model_dir}/${time_str}/update --mode export  --type update

# rm -rf ${model_dir}/ckpt_tmp



