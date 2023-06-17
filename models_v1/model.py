#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf
from common.ple_networks import PleNetworks

feat_map = {
    "u_snd_sys_cnt": "i_snd_sys_cnt",
    "u_snd_lnk_cnt": "i_snd_lnk_cnt",
    "u_rec_sys_cnt": "i_rec_sys_cnt",
    "u_rec_lnk_cnt": "i_rec_lnk_cnt",
    "u_price_sum": "i_price_sum",
    "u_ans_msg_rate": "i_ans_msg_rate",
    "u_sys_no_ans": "i_sys_no_ans",
    "u_sys_ans1": "i_sys_ans1",
    "u_establish_sess": "i_establish_sess",
    "u_valid_sess": "i_valid_sess",
    "u_deep_sess": "i_deep_sess",
    "u_last_deep_sess": "i_last_deep_sess",
    "u_ans_uid_cnt": "i_ans_uid_cnt",
    "u_rec_uid_cnt": "i_rec_uid_cnt",
    "u_sess_cnt": "i_sess_cnt",
    "u_ans_rate": "i_ans_rate"
}


def get_table_name(k):
    if k in ['u_v_ans_uid', 'u_v_deep_uid']:
        return 'i_uuid'
    if k in ['i_v_ans_uid', 'i_v_deep_uid']:
        return 'u_uuid'
    if k in feat_map:
        return feat_map[k]
    return k


class DnnModel:
    def __init__(self, emb_map, features, slot_list, is_training=True):
        super(DnnModel, self).__init__()
        self.is_training = is_training
        ######################label################################
        label1 = features["label1"]
        label1 = tf.cast(tf.reshape(label1, (-1,)), dtype=tf.float32)

        label2 = features["ui_hs"]
        label2 = tf.cast(tf.reshape(label2, (-1,)), dtype=tf.float32)
        label2 = tf.where(label2 >= 30, tf.ones_like(label2, tf.float32), tf.zeros_like(label2, tf.float32))

        label1 = tf.where(tf.logical_and(label1 < 1, label2 >= 30), tf.ones_like(label1, tf.float32), label1)
        self.labels = [label1, label2]
        # emb_map = {k: tf.keras.layers.Dropout(0.2)(emb_map[k]) for k in emb_map}
        ######################pop features不用特征################################

        ui_etype = emb_map.pop("ui_etype")
        slot_list.remove("ui_etype")
        input_emb = tf.concat([tf.multiply(tf.Variable(1.0, name=k), emb_map[k]) for k in slot_list], axis=1)
        ######################forward################################
        ple = PleNetworks(input_emb, 4, [4] * 2, [[128]])
        outs = ple.get_task_output()
        self.logits = []
        for i, out in enumerate(outs, start=1):
            logit = self.build_layers(tf.concat([out, ui_etype], axis=1), [256, 128, 1], "task%s" % i)
            logit = tf.clip_by_value(tf.reshape(logit, (-1,)), -15, 15)
            self.logits.append(logit)

        ######################################################
        self.probs = [tf.sigmoid(logit) for logit in self.logits]
        self.outputs = tf.concat([self.probs[0], self.probs[1]], axis=-1)
        self.loss = 0
        for i, (label, logit) in enumerate(zip(self.labels, self.logits), start=1):
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
            self.loss += loss
            tf.summary.scalar("loss%s" % i, loss)
        self.predictions = {
            "id": features["u_uuid"],
            "out": tf.concat(tf.concat([tf.reshape(x, [-1, 1]) for x in self.labels + self.probs], axis=1), axis=1)
        }

    def build_layers(self, inp, units, prefix, activation=None):
        act = "relu"
        layers = []
        for i, unit in enumerate(units):
            name = 'dnn_hidden_%s_%d' % (prefix, i)
            if i == len(units) - 1:
                act = activation
            #     dropout = None
            # else:
            #     dropout = tf.keras.layers.Dropout(0.5)
            layer = tf.keras.layers.Dense(
                units=unit, activation=act,
                kernel_initializer=tf.compat.v1.glorot_uniform_initializer(),
                bias_initializer=tf.compat.v1.glorot_uniform_initializer(),
                name=name
            )
            layers.append(layer)
            # if dropout is not None:
            #     layers.append(dropout)
        return tf.keras.Sequential(layers)(inp)