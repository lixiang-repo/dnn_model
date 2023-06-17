#!/usr/bin/env python
# coding=utf-8
import os
import datetime
import tensorflow as tf
from sklearn import metrics
from tensorflow.python.ops import io_ops
from tensorflow.python.training import checkpoint_management
from tensorflow.python.platform import gfile
from collections import defaultdict
from common.dataset import get_example_fmt, get_sequence_example_fmt

dirname = os.path.dirname(os.path.abspath(__file__))


def get_files(file_path, file_list, suffix=""):
    for file in os.listdir(file_path):
        if os.path.isdir(os.path.join(file_path, file)):
            get_files(os.path.join(file_path, file), file_list, suffix)
        else:
            file_list.append(os.path.join(file_path, file))

    return file_list if suffix == '' or suffix is None else list(filter(lambda x: x.endswith(suffix), file_list))


def write_donefile(time_str, model_type):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("./donefile") as f:
        donefile = list(map(lambda x: x.strip("\n"), f.readlines()))
    donefile.append("%s\t%s\t%s" % (time_str, model_type, now))
    with open("./donefile", "w") as wf:
        wf.write("\n".join(donefile))


def serving_input_receiver_dense_fn():
    tf.compat.v1.disable_eager_execution()
    feature_list = []
    with open("%s/../slot.conf" % dirname) as f:
        for line in f:
            if line.startswith("#"):
                continue
            feature_list.extend(line.strip("\n").split(":"))
    example_fmt = get_example_fmt()
    feature_spec = {}
    for k in example_fmt:
        dtype = example_fmt[k].dtype
        shape = (None,) + tuple(example_fmt[k].shape)
        feature_spec[k] = tf.compat.v1.placeholder(dtype, shape, name=k)

    return tf.compat.v1.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)


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


def _gauc(user_id, label, prob):
    uid_label_map = defaultdict(list)
    uid_prob_map = defaultdict(list)
    for i in range(len(user_id)):
        uid_label_map[user_id[i]].append(label[i])
        uid_prob_map[user_id[i]].append(prob[i])

    total_imp = 0
    total_auc = 0
    for uid in uid_label_map:
        try:
            imp = len(uid_label_map[uid])
            auc = imp * metrics.roc_auc_score(uid_label_map[uid], uid_prob_map[uid])
            total_imp += imp
            total_auc += auc
        except:
            pass
    return total_auc / (total_imp + 1e-10)


def get_metrics(df):
    metric_map = {}
    user_id = df.index
    y = df.iloc[:, :df.shape[1] // 2].values
    p = df.iloc[:, df.shape[1] // 2:].values
    assert y.shape == p.shape
    for i in range(y.shape[1]):
        auc = metrics.roc_auc_score(y[:, i], p[:, i])
        pcoc = p[:, i].mean() / y[:, i].mean()
        gauc = _gauc(user_id, y[:, i], p[:, i])
        mae = metrics.mean_absolute_error(y[:, i], p[:, i])
        real_ctr = y[:, i].mean()
        prob = p[:, i].mean()

        _metric_names = map(lambda x: "%s_%s" % (x, i), "auc, pcoc, gauc, mae, real_ctr, prob".split(", "))
        _metric_map = dict(zip(_metric_names, [auc, pcoc, gauc, mae, real_ctr, prob]))
        metric_map.update(_metric_map)

    return metric_map
