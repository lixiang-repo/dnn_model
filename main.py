import sys

import tensorflow as tf
import tensorflow_recommenders_addons as tfra
import pandas as pd
from absl import app, flags
from common.model_fn import model_fn
import os
import json
import time
from dateutil.parser import parse
import numpy as np

from common.dataset import input_fn
from common.utils import write_donefile, get_metrics, serving_input_receiver_dense_fn

flags.DEFINE_string('model_dir', "./ckpt", 'export_dir')
flags.DEFINE_string('export_dir', "./export_dir", 'export_dir')
flags.DEFINE_string('mode', "train", 'train or export')
flags.DEFINE_string('warm_path', '', 'warm start path')
flags.DEFINE_string('data_path', '/data/lixiang/data/matchmaking_girls_to_newboys_v1/', 'data path')
flags.DEFINE_string("type", "join", "join or update model")
# flags.DEFINE_string('time_format', '%Y%m%d/%H/%M', 'time format for training')
flags.DEFINE_string('time_format', '%Y%m%d', 'time format for training')
flags.DEFINE_string('time_str', '202305270059', 'training time str')
flags.DEFINE_float('lr', 1.0, 'lr dense train ')
FLAGS = flags.FLAGS

tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
task_type = tf_config.get('task', {}).get('type', "chief")
task_idx = tf_config.get('task', {}).get('index', 0)
ps_num = len(tf_config.get('cluster', {}).get('ps', []))
task_number = len(tf_config.get('cluster', {}).get('worker', [])) + 1
task_idx = task_idx + 1 if task_type == 'worker' else task_idx

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)


def train(estimator, slot=None):
    # ==========  执行任务  ========== #
    time_format = parse(FLAGS.time_str).strftime(FLAGS.time_format)
    file_pattern = "%s/%s" % (FLAGS.data_path, time_format)
    # filenames = ["/tf/data/data-tfrecord-train/part-r-00000.gz"]
    # with open("./train_list.txt") as f:
    #     file_pattern = [l.strip("\n") for l in f]
    # tf.compat.v1.logging.info("file_pattern>>>%s" % file_pattern)

    if FLAGS.mode == "train":
        # train_spec = tf.estimator.TrainSpec(
        #     input_fn=lambda: input_fn(file_pattern, task_number, task_idx, shuffle=True))
        # eval_spec = tf.estimator.EvalSpec(
        #     input_fn=lambda: input_fn(file_pattern.replace("gz", "_SUCCESS"), task_number, task_idx))
        # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        estimator.train(lambda: input_fn(file_pattern, task_number, task_idx, shuffle=True))
        if task_type == "chief" and int(task_idx) == 0:
            write_donefile(FLAGS.time_str, FLAGS.type)
    elif FLAGS.mode in ["eval", "feature_eval"]:
        estimator.evaluate(input_fn=lambda: input_fn(file_pattern, task_number, task_idx))
    elif FLAGS.mode in ["infer", "embeddings"]:
        outs = list(estimator.predict(input_fn=lambda: input_fn(file_pattern, task_number, task_idx)))
        print("outs>>>", outs[:10])
        df = pd.DataFrame(map(lambda x: x["out"], outs))
        df.index = map(lambda x: x["id"].decode("utf8"), outs)
        df.index.name = "id"

        if FLAGS.mode == "embeddings":
            df = df.reset_index().drop_duplicates(["id"] + list(range(9))).sort_values("id")
            df.to_csv("./embeddings.csv", sep="\t", index=False)
        else:
            metric_map = get_metrics(df)
            metric_map["slot"] = slot
            tf.compat.v1.logging.info("feature_eval>>>%s" % metric_map)
            df.to_csv("./pred.csv", sep="\t")
        # df = df.reset_index().drop_duplicates(["id"] + list(range(9))).sort_values("id")
        # df.to_csv("./embeddings.csv", sep="\t", index=False)

    elif FLAGS.mode == "dump":
        tf.compat.v1.logging.info("lx>>>%s" % estimator.get_variable_names())
        keys = estimator.get_variable_value("embeddings/embeddings_mht_1of1-keys")
        values = estimator.get_variable_value("embeddings/embeddings_mht_1of1-values")
        emb = np.concatenate((np.reshape(keys, [-1, 1]), values), axis=1)
        np.savetxt("%s/emb.txt" % FLAGS.model_dir, emb, fmt=['%d'] + ['%.6f'] * 9)
    elif FLAGS.mode == "preview":
        tf.compat.v1.logging.info("lx>>>%s" % estimator.get_variable_names())
        values = estimator.get_variable_value("embeddings/embeddings_mht_1of1-values")
        np.savetxt("%s/var.txt" % FLAGS.model_dir, values, fmt='%.6f')

    elif FLAGS.mode == "export" and task_type == "chief" and int(task_idx) == 0:
        from common.utils import serving_input_receiver_dense_fn
        tf.compat.v1.logging.info("export>>>%s" % FLAGS.model_dir)
        tfra.dynamic_embedding.enable_inference_mode()
        estimator.export_savedmodel(FLAGS.export_dir, lambda: serving_input_receiver_dense_fn())


def main(argv):
    del argv
    if FLAGS.type == "join":
        sys.exit(0)

    assert FLAGS.type in ["join", "update"]
    assert FLAGS.mode in ["train", "eval", "infer", "export", "embeddings", "feature_eval", "infer", "dump", "preview"]
    params = {
        "mode": FLAGS.mode,
        "warm_path": FLAGS.warm_path,
        "ps_num": ps_num,
        "type": FLAGS.type,
        "lr": FLAGS.lr,
        "slot": None
    }
    model_config = tf.estimator.RunConfig().replace(
        keep_checkpoint_max=1,
        save_checkpoints_secs=1200,
        log_step_count_steps=5000,
        save_summary_steps=10000,
        session_config=tf.compat.v1.ConfigProto(
            inter_op_parallelism_threads=os.cpu_count() // 2,
            intra_op_parallelism_threads=os.cpu_count() // 2))

    # Save checkpoints
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=params, config=model_config)

    t1 = time.time()
    if FLAGS.mode == "export" and task_type == "chief" and task_idx == 0:
        estimator.export_saved_model(FLAGS.export_dir, lambda: serving_input_receiver_dense_fn())
    elif FLAGS.mode == "feature_eval":
        t1 = time.time()
        tf.compat.v1.logging.info("feature_eval>>>start")
        params["slot"] = "base"
        train(estimator, "base")
        t2 = time.time()
        tf.compat.v1.logging.info("waste time>>>%s>>>%s>>>%s mins" % (FLAGS.type, FLAGS.time_str, (t2 - t1) / 60))

        slot_list = "user_id,hour".split(",")
        slot_list = slot_list.split(":")
        for slot in slot_list:
            params["slot"] = slot
            estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir,
                                               params=params, config=model_config)
            train(estimator, slot)
            t2 = time.time()
            tf.compat.v1.logging.info(
                "waste time>>>%s>>>%s>>>%s>>>%s mins" % (slot, FLAGS.type, FLAGS.time_str, (t2 - t1) / 60))

    else:
        train(estimator)

    t2 = time.time()
    tf.compat.v1.logging.info("waste time>>>%s>>>%s>>>%s mins" % (FLAGS.type, FLAGS.time_str, (t2 - t1) / 60))
    tf.compat.v1.logging.info("return>>>%s>>>%s>>>%s" % (FLAGS.mode, task_type, task_idx))


if __name__ == "__main__":
    app.run(main)
