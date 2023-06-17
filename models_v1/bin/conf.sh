start_date=202208232359
end_date=202505080259
delta=1
time_format=%Y%m%d/%H/*/*.gz
model_dir="/nas/lixiang/data/tfra_social_rank_v1"
nas_path="/nas/yuchengchao/online_data_matchmaker_v2"

############################################################################################################
############################################################################################################
donefile=${model_dir}/donefile
mkdir -p ${model_dir} > /dev/null
touch ${donefile}