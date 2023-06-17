import json
import os

import tensorflow as tf

import tensorflow_recommenders_addons as tfra
import tensorflow_recommenders_addons.dynamic_embedding as de
import numpy as np
import pandas as pd

from absl import app
from absl import flags

from tensorflow.python.ops import io_ops
from tensorflow.python.training import checkpoint_management
from tensorflow.python.platform import gfile

flags.DEFINE_string('model_dir', "./test", 'export_dir')
flags.DEFINE_string('export_dir', "./export_dir", 'export_dir')
flags.DEFINE_string('mode', "train", 'train or export')
flags.DEFINE_string('warm_path', '', 'warm start path')
flags.DEFINE_string('data_path', '/tf/data/matchmaking_girls_to_newboys_v1/', 'data path')
flags.DEFINE_string("type", "join", "join or update model")
# flags.DEFINE_string('time_format', '%Y%m%d/%H/%M', 'time format for training')
flags.DEFINE_string('time_format', '%Y%m%d', 'time format for training')
flags.DEFINE_string('time_str', '202305270059', 'training time str')
flags.DEFINE_float('lr', 1.0, 'lr dense train ')
flags.DEFINE_string('var', "", ' ')

flags.DEFINE_string('task_type', "", ' ')
flags.DEFINE_integer('task_idx', 0, ' ')
FLAGS = flags.FLAGS

tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
task_type = tf_config.get('task', {}).get('type', "chief")
task_idx = tf_config.get('task', {}).get('index', 0)
ps_num = len(tf_config.get('cluster', {}).get('ps', []))
task_number = len(tf_config.get('cluster', {}).get('worker', [])) + 1
task_idx = task_idx + 1 if task_type == 'worker' else task_idx


def restore_tfra_variable(ckpt_path, variable):
    mhts = variable._tables
    key_tensor_names = []
    value_tensor_names = []
    for mht in mhts:
        assert len(mht.saveable.specs) == 2
        key_tensor_names.append(mht.saveable.specs[0].name)
        value_tensor_names.append(mht.saveable.specs[1].name)

    latest_ckpt = _get_checkpoint_filename(ckpt_path)
    restore_op = io_ops.restore_v2(latest_ckpt,
                                   key_tensor_names + value_tensor_names, [""] *
                                   len(key_tensor_names + value_tensor_names),
                                   ([tf.int64] * len(key_tensor_names)) +
                                   ([tf.float32] * len(value_tensor_names)))

    key_tensor = restore_op[:len(key_tensor_names)]
    value_tensor = restore_op[len(key_tensor_names):]

    mht_restore_ops = []
    index = 0
    for mht in mhts:
        mht_restore_ops.append(
            mht.saveable.restore((key_tensor[index], value_tensor[index]), None))
        index += 1
    return mht_restore_ops


class RestoreTfraVariableHook(tf.estimator.SessionRunHook):

    def __init__(self, ckpt_path, variables):
        self._ckpt_path = ckpt_path
        self._variables = variables

    def begin(self):
        self._restore_ops = []
        for var in self._variables:
            self._restore_ops.extend(restore_tfra_variable(self._ckpt_path, var))

    def after_create_session(self, session, coord):
        session.run(self._restore_ops)


def _get_checkpoint_filename(ckpt_dir_or_file):
    if gfile.IsDirectory(ckpt_dir_or_file):
        return checkpoint_management.latest_checkpoint(ckpt_dir_or_file)
    return ckpt_dir_or_file


def input_fn():
    movie_id = np.random.randint(0, 1000, (100000, 1))
    user_id = np.random.randint(0, 1000, (100000, 1))
    tf.compat.v1.logging.info("size>>>%s" % len(set(list(movie_id[:, 0])).union(list(user_id[:, 0]))))
    z = movie_id + user_id
    dataset = {
        "movie_id": movie_id,
        "user_id": user_id,
        "user_rating": 1 / (1 + np.exp(-z))
    }
    dataset = tf.data.Dataset.from_tensor_slices(dataset).repeat(20).batch(256)
    if task_number > 1:
        dataset = dataset.shard(task_number, task_idx)

    return dataset


def model_fn(features, labels, mode, params):
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    embedding_size = 9
    movie_id = features["movie_id"]
    u_uuid = features["user_id"]
    label1 = label2 = label3 = tf.cast(tf.reshape(features.pop("user_rating"), (-1,)), tf.float32)
    ######################pop features不用特征################################
    # features.pop('ui_hs')

    ######################slot################################
    slot_list = ["movie_id", "user_id"]
    tf.compat.v1.logging.debug("slot_list>>>%s" % ":".join(slot_list))
    ######################devide################################
    if params["type"] == "join" or not is_training:
        initializer = tf.compat.v1.initializers.zeros()
    else:
        initializer = tf.keras.initializers.RandomNormal(-1, 1)
    if params["ps_num"] > 0:
        ps_list = ["/job:ps/replica:0/task:{}/CPU:0".format(i) for i in range(params["ps_num"])]
    else:
        ps_list = ["/job:localhost/replica:0/task:0/CPU:0"] * params["ps_num"]  ##单机，分布式注释
    tf.compat.v1.logging.info("ps_list>>>%s" % ps_list)
    tf.compat.v1.logging.info("strategy>>>%s" % tf.compat.v1.distribute.get_strategy())
    # with tf.name_scope("dnn"):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Dense(3)
    ])

    ######################dnn################################
    feature_list = [features[k] for k in slot_list]
    ids = tf.concat(feature_list, axis=1)  # [None, 1] concat
    tf.compat.v1.logging.debug("ids_shape>>>%s" % ids.shape)
    # with tf.name_scope("embedding"):
    embeddings = tfra.dynamic_embedding.get_variable(
        name="embeddings",
        dim=embedding_size,
        devices=ps_list,
        # trainable=params["type"] == "update",
        initializer=initializer)
    ######################lookup################################
    emb_shape = tf.concat([tf.shape(ids), [embedding_size]], axis=0)
    id_val, id_idx = tf.unique(tf.reshape(ids, (-1,)))
    unique_embs, trainable_wrapper = de.embedding_lookup(embeddings, id_val, return_trainable=True, name="lookup")
    # tf.compat.v1.add_to_collections(de.GraphKeys.DYNAMIC_EMBEDDING_VARIABLES, embeddings)
    flat_emb = tf.gather(unique_embs, id_idx)
    emb_lookuped = tf.reshape(flat_emb, emb_shape)
    ######################lookup end################################
    num_or_size_splits = [features[k].shape[1] for k in slot_list]
    emb_splits = tf.split(emb_lookuped, num_or_size_splits, axis=1)
    emb_map = dict(zip(slot_list, emb_splits))
    for k in slot_list:
        emb_map[k] = tf.reduce_sum(emb_map[k], axis=1)
        tf.compat.v1.logging.debug("emb_shape>>>%s>>>%s" % (k, emb_map[k].shape))
    input_emb = tf.concat([emb_map[k] for k in slot_list], axis=1)

    ######################forward################################
    logits = [tf.reshape(x, (-1,)) for x in tf.split(model(input_emb), [1, 1, 1], axis=1)]

    loss_list = [
        tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(label, logits[i])) for i, label in
        zip(range(3), [label1, label2, label3])
    ]
    loss = sum(loss_list)
    for i in range(3):
        tf.summary.scalar("loss%s" % i, loss_list[i])

    # predictions = {"uid": tf.reshape(tf.concat(feature_list, 0), (-1,)), "out": flat_emb}
    probs = [tf.sigmoid(logits[i]) for i in range(3)]

    ######################predictions################################
    predictions = tf.concat([tf.reshape(tf.cast(x, tf.float32), [-1, 1]) for x in probs + [label1, label2, label3]], 1)
    predictions = {"id": u_uuid, "out": predictions}
    # predictions = {"id": tf.reshape(ids, (-1,)), "out": flat_emb}
    ######################metrics################################
    loggings = {"%s_emb_size" % params["type"]: embeddings.size(), "loss": loss}
    eval_metric_ops = {}
    tf.compat.v1.train.init_from_checkpoint("202306010259/join", {
        "embeddings/embeddings_mht_1of1-keys": "embeddings/",
        "embeddings/embeddings_mht_1of1-values": "embeddings/"
    })
    embeddings._get_variable_list
    ######################train################################
    if mode == tf.estimator.ModeKeys.TRAIN:
        ######################optimizer################################
        from tensorflow_estimator.python.estimator.canned import optimizers
        # sparse_opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
        sparse_opt = optimizers.get_optimizer_instance("SGD", learning_rate=0.01)
        sparse_opt = tfra.dynamic_embedding.DynamicEmbeddingOptimizer(sparse_opt)
        # dense_opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
        dense_opt = optimizers.get_optimizer_instance("SGD", learning_rate=0.001 * params["lr"])

        trainable_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        dense_vars = [var for var in trainable_variables if not var.name.startswith("lookup")]
        sparse_vars = [var for var in trainable_variables if var.name.startswith("lookup")]
        global_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)

        tf.compat.v1.logging.debug("dense_vars>>>%s" % dense_vars)
        tf.compat.v1.logging.debug("sparse_vars>>>%s>>>%s" % (id(trainable_wrapper), [id(x) for x in sparse_vars]))
        tf.compat.v1.logging.debug("global_vars>>>%s" % global_variables)
        tf.compat.v1.logging.debug("trainable_de_variables>>>%s" % tf.compat.v1.get_collection(
            de.GraphKeys.TRAINABLE_DYNAMIC_EMBEDDING_VARIABLES))
        tf.compat.v1.logging.debug("#" * 100)

        global_step = tf.compat.v1.train.get_or_create_global_step()
        dense_op = dense_opt.minimize(loss, global_step=global_step, var_list=dense_vars)
        if params["type"] == "join":
            train_op = dense_op
        elif params["type"] == "update":
            sparse_op = sparse_opt.minimize(loss, global_step=global_step, var_list=sparse_vars)
            train_op = tf.group(dense_op, sparse_op)
        else:
            raise RuntimeError("Unsupport type", params["type"])

        log_hook = tf.compat.v1.estimator.LoggingTensorHook(loggings, every_n_iter=500)
        ######################WarmStartHook################################
        training_chief_hooks = None
        if params["warm_path"]:
            tf.compat.v1.logging.info("train warm start>>>%s" % params["warm_path"])
            # restore_hook = de.WarmStartHook(params["warm_path"], [".*emb.*"])
            restore_hook = RestoreTfraVariableHook(params["warm_path"], [embeddings])
            training_chief_hooks = [restore_hook]
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions,
                                          loss=loss,
                                          train_op=train_op,
                                          training_hooks=[log_hook],
                                          training_chief_hooks=training_chief_hooks)
    ######################infer################################
    elif mode in [tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL]:
        _hooks = None
        if params["warm_path"]:
            tf.compat.v1.logging.info("infer warm start>>>%s" % params["warm_path"])
            # restore_hook = de.WarmStartHook(params["warm_path"], [embeddings])
            restore_hook = RestoreTfraVariableHook(params["warm_path"], [embeddings])
            _hooks = [restore_hook]
        outputs = tf.multiply(probs[0], probs[1])
        export_outputs = {
            "pred": tf.compat.v1.estimator.export.PredictOutput(outputs)
        }
        return tf.estimator.EstimatorSpec(mode, predictions, loss, export_outputs=export_outputs,
                                          prediction_hooks=_hooks, evaluation_hooks=_hooks,
                                          eval_metric_ops=eval_metric_ops)


def serving_input_receiver_dense_fn():
    # input_spec = {
    #     "movie_id": tf.constant([1], tf.int64),
    #     "user_id": tf.constant([1], tf.int64),
    #     "user_rating": tf.constant([1.0], tf.float32)
    # }
    tf.compat.v1.disable_eager_execution()

    input_spec = {
        "movie_id": tf.compat.v1.placeholder(tf.int64, (None,)),
        "user_id": tf.compat.v1.placeholder(tf.int64, (None,)),
        "user_rating": tf.compat.v1.placeholder(tf.float32, (None,)),
    }
    return tf.estimator.export.build_raw_serving_input_receiver_fn(input_spec)


def export_for_serving(model_dir, export_dir):
    model_config = tf.estimator.RunConfig(log_step_count_steps=100,
                                          save_summary_steps=100,
                                          save_checkpoints_steps=100,
                                          save_checkpoints_secs=None)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=model_dir,
                                       params={"ps_num": ps_num},
                                       config=model_config)

    estimator.export_saved_model(export_dir, serving_input_receiver_dense_fn())


def train(model_dir):
    params = {
        "mode": FLAGS.mode,
        "warm_path": FLAGS.warm_path,
        "ps_num": ps_num,
        "type": FLAGS.type,
        "lr": FLAGS.lr,
        "slot": None
    }
    model_config = tf.estimator.RunConfig(log_step_count_steps=100,
                                          save_summary_steps=100,
                                          save_checkpoints_steps=100,
                                          save_checkpoints_secs=None,
                                          keep_checkpoint_max=1)
    # Save checkpoints
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       model_dir=model_dir,
                                       params=params,
                                       config=model_config)

    # ==========  执行任务  ========== #
    if FLAGS.mode == "train":
        # train_spec = tf.estimator.TrainSpec(input_fn=input_fn)
        # eval_spec = tf.estimator.EvalSpec(input_fn=input_fn)
        # tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
        estimator.train(input_fn)
    elif FLAGS.mode == "eval":
        estimator.evaluate(input_fn=input_fn)
    elif FLAGS.mode == "infer":
        outs = list(estimator.predict(input_fn=input_fn, predict_keys=None))
        df = pd.DataFrame(map(lambda x: x["out"], outs))
        df.index = map(lambda x: x["id"], outs)
        df.index.name = "id"
        print(df)
        df = df.reset_index().drop_duplicates(["id"] + list(range(1))).sort_values("id")

        df.to_csv("./result.csv", sep="\t", index=False)
    elif FLAGS.mode == "dump":
        tf.compat.v1.logging.info("lx>>>%s" % estimator.get_variable_names())
        keys = estimator.get_variable_value("embeddings/embeddings_mht_1of1-keys")
        values = estimator.get_variable_value("embeddings/embeddings_mht_1of1-values")
        emb = np.concatenate((np.reshape(keys, [-1, 1]), values), axis=1)
        np.savetxt("%s/emb.txt" % FLAGS.model_dir, emb, fmt=['%d'] + ['%.6f'] * 9)

    elif FLAGS.mode == "preview":
        variables = tf.compat.v1.train.list_variables("test")
        tf.compat.v1.logging.info("variables>>>%s" % variables)

        estimator.train(input_fn)

        tf.compat.v1.logging.info("lx>>>%s" % estimator.get_variable_names())
        keys = estimator.get_variable_value("embeddings/embeddings_mht_1of1-keys")
        values = estimator.get_variable_value("embeddings/embeddings_mht_1of1-values")
        emb = np.concatenate((np.reshape(keys, [-1, 1]), values), axis=1)
        np.savetxt("%s/emb.txt" % FLAGS.model_dir, emb, fmt=['%d'] + ['%.6f'] * 9)
    elif FLAGS.mode == "var":
        tf.compat.v1.logging.info("lx>>>%s" % estimator.get_variable_names())
        values = estimator.get_variable_value(FLAGS.var)
        np.savetxt("%s/var.txt" % FLAGS.model_dir, values, fmt='%.6f')


def main(argv):
    del argv

    train(FLAGS.model_dir)

    if FLAGS.mode == "export" and task_type == "chief" and int(task_idx) == 0:
        tfra.dynamic_embedding.enable_inference_mode()
        export_for_serving(FLAGS.model_dir, FLAGS.export_dir)

    tf.compat.v1.logging.info("done>>>")


if __name__ == "__main__":
    app.run(main)
