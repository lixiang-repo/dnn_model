import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
tf = tf.compat.v1
from tqdm import tqdm
import random
from common import metrics

rand = random.random()


labels = np.random.randint(0, 2, (10000, 1))
z = labels + (1.0 + rand) * np.random.random((10000, 1))
predictions = 1.0 / (1 + np.exp(-z))
n_batches = len(labels)
print("batchs", n_batches)

graph = tf.Graph()
with graph.as_default():
    # Placeholders to take in batches onf data
    tf_label = tf.placeholder(dtype=tf.int64, shape=[None])
    tf_prediction = tf.placeholder(dtype=tf.float32, shape=[None])

    # Define the metric and update operations
    with tf.name_scope("lx_metric"):
        tf_metric1, tf_metric_update1 = tf.metrics.mean(tf_label, name="my_metric1")
        tf_metric2, tf_metric_update2 = tf.metrics.mean(tf_prediction, name="my_metric2")
        tf_metric = tf.div_no_nan(tf_metric2, tf_metric1)
        tf_metric_update = math_ops.div_no_nan(tf_metric_update2, math_ops.maximum(tf_metric_update1, 0), name='update_op')
        # tf_metric_update = tf.group(tf_metric_update1, tf_metric_update2)

        sum_t, sum_op = metrics.sum_op(tf.ones_like(tf_label, dtype=tf.float32))

    # Isolate the variables stored behind the scenes by the metric operation
    running_vars1 = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="lx_metric")
    running_vars2 = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric2")

    # Define initializer to initialize/reset running variables
    running_vars_initializer = tf.variables_initializer(var_list=running_vars1 + running_vars2)

    print("running_vars>>>", running_vars1, running_vars2)

with tf.Session(graph=graph) as session:
    session.run(tf.global_variables_initializer())

    # initialize/reset the running variables
    session.run(running_vars_initializer)

    for i in tqdm(range(n_batches)):
        # Update the running variables on new batch of samples
        feed_dict = {tf_label: labels[i], tf_prediction: predictions[i]}
        update_op = session.run([tf_metric_update], feed_dict=feed_dict)
        res = session.run([sum_t, sum_op, tf_metric] + running_vars1 + running_vars2, feed_dict=feed_dict)
        print("lx>>>", update_op, res)

    # Calculate the score
    res = session.run([tf_metric] + running_vars1 + running_vars2, feed_dict=feed_dict)
    print("SCORE>>>", res)

