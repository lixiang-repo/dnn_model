#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && cd .. && pwd)"
main_py="${code_dir}/main.py"
sh ${code_dir}/bin/stop.sh

init() {
  if ! test -e donefile; then
    touch donefile
  fi
  if ! test -e ckpt; then
    mkdir -p ckpt/join
    mkdir -p ckpt/update
  fi
}

data_path="./data"
#data_path="./"
model_dir="/data/lixiang/recommenders-addons/demo/dynamic_embedding/data"
mkdir -p ${model_dir} && cd ${model_dir}
start_date=202305080259
end_date=
init

i=0
while [ "${last_time}" != "202505080259" ]; do
  last_time=$(awk '{print $1}' donefile | tail -1)
  if [ "${last_time}" = "" ]; then
    last_time=${start_date}
    warm_path=""
    echo "start_date>>>${last_time}"
  else
    warm_path="${model_dir}/${last_time}/update"
  fi
  hour_str=$(python3 -c "from dateutil.parser import parse;import datetime;print((parse(str("${last_time}")) + datetime.timedelta(days=1)).strftime('%Y%m%d/%H/%M'))")
  time_str=$(python3 -c "from dateutil.parser import parse;import datetime;print((parse(str("${last_time}")) + datetime.timedelta(days=1)).strftime('%Y%m%d%H%M'))")

  if test -e ${data_path}/${hour_str}/_SUCCESS; then
#    echo "start join_model>>>${time_str}>>>${data_path}/${hour_str}"
#    python3 ${main_py} --model_dir "${model_dir}/ckpt/join" --warm_path "${warm_path}" --mode train --type "join" --time_str "${time_str}" --time_format "%Y%m%d/*/*/*.gz" || exit 1
#    echo "end join_model>>>${time_str}>>>${data_path}/${hour_str}"
    ############################################################
    echo "start update_model>>>${time_str}>>>${data_path}/${hour_str}"
    python3 ${main_py} --model_dir "${model_dir}/ckpt/update" --mode train --type "update" --data_path "${data_path}" --time_str "${time_str}" --time_format "%Y%m%d/*/*/*.gz" || exit 2
    echo "end update_model>>>${time_str}>>>${data_path}/${hour_str}"

    #backup
    cp -r ckpt ${time_str}
    rm -rf ${time_str}/*/events.out.tfevents*
    rm -rf ${time_str}/*/eval
  else
    sleep 0.001
  fi
  let i=$i+1
  let j=$i%10
  if [ $j -eq 0 ]; then
    rm -rf ckpt/*/events.out.tfevents*
    rm -rf ckpt/*/eval
  fi
done
