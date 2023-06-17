#!/usr/bin/env bash
code_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && cd .. && pwd)"
. ${code_dir}/bin/conf.sh && cd ${model_dir}

code="""
import json, os
tf_config = json.loads('${TF_CONFIG}' or '{}')
print(tf_config['task']['type'])
"""

if [ ! $(python -c "${code}") = "chief" ];then
    sleep 1000000000
fi

TF_CONFIG='{}'

# sh -x ${code_dir}/bin/stop.sh
# python3 ${code_dir}/main.py --model_dir ${model_dir}/ckpt/update --mode train --type update \
#     --data_path /nas/yuchengchao/online_data_matchmaker_v2 --time_str 202307251331 \
#     --file_list ${model_dir}/train.txt


cd ${code_dir}/bin
sh -x stop.sh && sh -x run.sh


# python3 ${code_dir}/main.py --model_dir ${model_dir}/ckpt/update --mode train --type update \
#     --data_path /nas/yuchengchao/online_data_matchmaker_v2 --time_str 202307062359 \
#     --file_list /nas/lixiang/data/taqu_social_rank_ple_v6_eval/train_list.txt



# sh ${code_dir}/bin/stop.sh
# python3 ${code_dir}/main.py --model_dir ${model_dir}/202307062359/update \
#     --mode eval --type update --data_path ${nas_path} --slot "base" \
#     --file_list /nas/lixiang/data/taqu_social_rank_ple_v6_eval/eval_list.txt
# for slot in `cat ${code_dir}/slot.conf|grep -v label|grep -v "#"|awk '{print $1}'`
# do
#     echo "evalslot>>> ${slot}"

#     sh ${code_dir}/bin/stop.sh
#     python3 ${code_dir}/main.py --model_dir ${model_dir}/202307062359/update \
#         --mode eval --type update --data_path ${nas_path} --slot "${slot}" \
#         --file_list /nas/lixiang/data/taqu_social_rank_ple_v6_eval/eval_list.txt
# done

