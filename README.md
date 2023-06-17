# A distributed training demo for `tfra.dynamic_embedding`:

- model: DNN
- Running API: using estimator APIs

## start train:
By default, this shell will start a train task with 1 PS and 1 workers and 1 chief on local machine.
sh run.sh


## start export for serving:
By default, this shell will start a export for serving task with 1 PS and 1 workers and 1 chief on local machine.
sh export.sh

## stop.train
run sh stop.sh