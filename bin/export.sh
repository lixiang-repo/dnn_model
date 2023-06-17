#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && cd .. && pwd)"
main_py="${code_dir}/main.py"
sh ${code_dir}/bin/stop.sh

base_dir="./data"
cd ${base_dir}
time_str=$(awk '{print $1}' donefile | tail -1)
if [ ! "$1" = "" ]; then
  time_str="$1"
fi
echo "export_time_str>>>${time_str}"

model_dir="${base_dir}/${time_str}/update/"
cp -r ${export_dir} ${export_dir}_tmp


#python3 main.py --model_dir "${time_str}_tmp/join" --warm_path "${time_str}_tmp/update" --mode train --type "join" --time_str "${time_str}" --lr 0.0 --time_format "%Y%m%d/%H/%M"
python3 ${main_py} --model_dir ${model_dir} --mode export  --type update

#rm -rf ${time_str}_tmp



