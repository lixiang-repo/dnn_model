import tensorflow as tf
from common.ple_networks import PleNetworks

feat_map = {
    "a_fea": "b_fea",
}


def get_table_name(k):
    return k


class DnnModel:
    def __init__(self, emb_map, features, slot_list, is_training=True):
        super(DnnModel, self).__init__()
        if not is_training:
            self.initializer = tf.keras.initializers.Zeros()

        else:
            self.initializer = tf.compat.v1.glorot_uniform_initializer()
        ######################label################################
        label1 = tf.reshape(features.pop("label1"), (-1,))

        ui_hs = tf.reshape(features.pop("label2"), (-1,))
        label2 = tf.cast(ui_hs, dtype=tf.float32)
        label2 = tf.where(label2 >= 30, tf.ones_like(label2, tf.float32), tf.zeros_like(label2, tf.float32))

        label3 = tf.cast(ui_hs, dtype=tf.float32)
        label3 = tf.where(label3 >= 60, tf.ones_like(label3, tf.float32), tf.zeros_like(label3, tf.float32))

        label1 = tf.where(tf.logical_and(label1 < 1, label2 >= 30), tf.ones_like(label1, tf.float32), label1)
        self.labels = [label1, label2, label3]
        ######################pop features不用特征################################

        ui_etype = emb_map.pop("ui_etype")
        slot_list.remove("ui_etype")
        input_emb = tf.concat([tf.multiply(tf.Variable(1.0, name=k), emb_map[k]) for k in slot_list], axis=1)
        ######################forward################################
        ple = PleNetworks(input_emb, 4, [4] * 3, [[128]])
        outs = ple.get_task_output()
        self.logits = []
        for i, out in enumerate(outs, start=1):
            logit = self.build_layers(tf.concat([out, ui_etype], axis=1), [256, 128, 1], "task%s" % i)
            logit = tf.clip_by_value(tf.reshape(logit, (-1,)), -15, 15)
            self.logits.append(logit)
        self.probs = [tf.sigmoid(logit) for logit in self.logits]
        self.outputs = tf.concat([tf.reshape(x, [-1, 1]) for x in self.probs], axis=1)

        self.predictions = {
            "id": features["u_uuid"],
            "out": tf.concat(tf.concat([tf.reshape(x, [-1, 1]) for x in self.labels + self.probs], axis=1), axis=1)
        }

        self.loss_func(features)

    def loss_func(self, features):
        loss_list = [
            tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit)
            ) for label, logit in zip(self.labels, self.logits)
        ]
        self.loss = sum(loss_list)
        for i, _loss in enumerate(loss_list, start=1):
            tf.summary.scalar("loss%s" % i, _loss)



    def build_layers(self, inp, units, prefix, activation=None):
        act = "relu"
        layers = []
        for i, unit in enumerate(units):
            name = 'dnn_hidden_%s_%d' % (prefix, i)
            if i == len(units) - 1:
                act = activation
            layer = tf.keras.layers.Dense(
                units=unit, activation=act,
                kernel_initializer=self.initializer,
                bias_initializer=self.initializer,
                name=name
            )
            layers.append(layer)
        return tf.keras.Sequential(layers)(inp)
