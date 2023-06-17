#!/usr/bin/env python
# coding=utf-8
import tensorflow as tf

tf = tf.compat.v1
import re
from collections import OrderedDict
import os

dirname = os.path.dirname(os.path.abspath(__file__))


def _parse_type(type_str):
    type_str = type_str.upper()
    if type_str == 'INT':
        return tf.int64
    elif type_str == 'BIGINT':
        return tf.int64
    elif type_str == 'DOUBLE':
        return tf.float64
    elif type_str == 'FLOAT':
        return tf.float32
    elif type_str == 'STRING':
        return tf.string
    else:
        arr_re = re.compile("ARRAY<(.*)>")
        t = arr_re.findall(type_str)
        if len(t) == 1:
            return _parse_type(t[0])
        raise TypeError("Unsupport type", type_str)


def get_slot_list():
    slots = []
    with open("%s/../slot.conf" % dirname) as f:
        for l in f:
            if l.startswith("#") or l.startswith("label"):
                continue
            slots.extend(re.split(" +", l.strip("\n"))[0].split("-"))
    return sorted(list(set(slots)))


def get_example_fmt():
    example_fmt = OrderedDict()
    slots = get_slot_list()
    with open("%s/../feature.conf" % dirname) as f:
        for line in f:
            if line.startswith("#"):
                continue
            name, type_str = re.split(" +", line.strip("\n"))[:2]
            # if name not in slots:
            #     continue
            if "ARRAY" in line:
                example_fmt[name] = tf.io.FixedLenFeature([50], _parse_type(type_str))
            else:
                example_fmt[name] = tf.io.FixedLenFeature((), _parse_type(type_str))
    return example_fmt


def get_sequence_example_fmt():
    ctx_fmt, seq_fmt = OrderedDict(), OrderedDict()
    slots = get_slot_list()
    with open("%s/../feature.conf" % dirname) as f:
        for line in f:
            if line.startswith("#"):
                continue
            name, type_str = re.split(" +", line.strip("\n"))[:2]
            # if name.startswith("label"):
            #     type_str = "FLOAT"
            # if name not in slots:
            #     continue
            if "@ctx" in line:
                if "ARRAY" in line:
                    ctx_fmt[name] = tf.io.FixedLenFeature([50], _parse_type(type_str))
                else:
                    ctx_fmt[name] = tf.io.FixedLenFeature((), _parse_type(type_str))
            elif "@seq" in line:
                if "ARRAY" in line:
                    seq_fmt[name] = tf.io.FixedLenSequenceFeature([50], _parse_type(type_str))
                else:
                    seq_fmt[name] = tf.io.FixedLenSequenceFeature((), _parse_type(type_str))
            else:
                raise TypeError("not in @ctx and @seq")

    return ctx_fmt, seq_fmt


def input_fn(filenames, task_number=1, task_idx=0, shuffle=False, epochs=1, batch_size=128):
    example_fmt = get_example_fmt()

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if task_number > 1:
        dataset = dataset.shard(task_number, task_idx)

    dataset = tf.data.TFRecordDataset(dataset, compression_type="GZIP")
    dataset = dataset.repeat(epochs).prefetch(10000)

    # Randomizes input using a window of 256 elements (read into memory)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)

    # epochs from blending together.
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda x: tf.io.parse_example(x, features=example_fmt), 10)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())

    return dataset.make_one_shot_iterator().get_next()


if __name__ == "__main__":
    tf.disable_v2_behavior()
    # from flags import FLAGS
    file_pattern = "/nas/lixiang/data/matchmaking_girls_to_newboys_v1/20230613/02/59/part-r-00000.gz"
    next_element = input_fn(file_pattern, shuffle=False)

    tmp = []
    with tf.Session() as sess:
        # sess.run(fetches=tf.global_variables_initializer())
        try:
            while True:
                # 通过session每次从数据集中取值
                serialized_examples = sess.run(fetches=next_element)
                print(serialized_examples)
                break
                # sess.run(fetches=train, feed_dict={x: image, y_: label})
                # if i % 100 == 0:
                #     train_accuracy = sess.run(fetches=accuracy, feed_dict={x: image, y_: label})
                #     print(i, "accuracy=", train_accuracy)
                # i = i + 1
        except tf.errors.OutOfRangeError:
            print("end!")
        print(tmp[0])
    # for features, labels in dataset:
    # for k in features:
    #     print(k, features[k])
    # break
    #
    # options = tf.io.TFRecordOptions(compression_type="GZIP")
    # tfrecord_path = "/tf/part-r-00000.gz"
    # for serialized_example in tf.io.tf_record_iterator(tfrecord_path, options=options):
    #     print(serialized_example)
    #     break

