#!/usr/bin/env python
# coding=utf-8
import json
import os

import tensorflow as tf

import numpy as np
import pandas as pd

from absl import app
from absl import flags
import glob

from tensorflow.python.ops import io_ops
from tensorflow.python.training import checkpoint_management
from tensorflow.python.platform import gfile


if __name__ == "__main__":
    with open("/nas/lixiang/data/taqu_social_rank_ple_v6_eval/eval_list.txt") as f:
        filenames = [l.strip("\n") for i, l in enumerate(f)]
    
    # dataset = tf.data.Dataset.list_files(file_pattern)#.shard(4, 4)
    dataset = tf.data.TFRecordDataset(filenames, compression_type="GZIP")
    next_element = dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        try:
            while True:
                # 通过session每次从数据集中取值
                res = sess.run(fetches=next_element)
                print("res>>>", res)#.decode("utf8"))
                # sess.run(fetches=train, feed_dict={x: image, y_: label})
                # if i % 100 == 0:
                #     train_accuracy = sess.run(fetches=accuracy, feed_dict={x: image, y_: label})
                #     print(i, "accuracy=", train_accuracy)
                # i = i + 1
        except tf.errors.OutOfRangeError:
            print("end!")