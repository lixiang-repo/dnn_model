# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
        tensor: A tf.Tensor to check the rank of.
        expected_rank: Python integer or list of integers, expected rank.
        name: Optional name of the tensor for the error message.

    Raises:
        ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


class Experts(object):
    def __init__(self, input_tensor, expert_num, scope,
                 expert_units_list,
                 activation=tf.nn.relu, use_bias=True,
                 initializer=tf.compat.v1.glorot_uniform_initializer(),
                 regularizer=None, dtype=tf.float32):
        """Constructor for Experts
        Args:
            input_tensor: Float Tensor of shape [batch_size, input_embedding_size].
            expert_num: Int scalar, expert number of each task.
            scope: Expert scope.
            expert_units_list: A list of int scalar.
            activation: Activation function (callable). Set it to None to maintain a linear activation.
            use_bias: Boolean, whether the layer uses a bias.
            initializer: Initializer function for the experts weight matrix.
            regularizer: Regularizer function for the experts weight matrix.
            dtype: type passed to get_variable.
            reuse: Whether to reuse the variable.
        """
        self._expert_num = expert_num

        self._experts_layer_list = []
        with tf.compat.v1.variable_scope('%s' % scope):
            for i in range(self._expert_num):
                self._experts_layer_list.append([])
                for j, expert_units in enumerate(expert_units_list):
                    experts_layer = tf.keras.layers.Dense(units=expert_units,
                                                          activation=activation,
                                                          name='expert_%d/dense_%d' % (i, j),
                                                          use_bias=use_bias,
                                                          kernel_initializer=initializer,
                                                          kernel_regularizer=regularizer,
                                                          dtype=dtype,
                                                          )

                    self._experts_layer_list[i].append(experts_layer)

                    # last layer units
                    self._expert_units = expert_units

            self._experts_output = self._compute_experts_output(input_tensor)

    def _compute_experts_output(self, input_tensor):
        """Compute experts output
        Args:
            input_tensor:  Float Tensor of shape [batch_size, input_embedding_size].
        Returns:
            experts_output: experts output shape [batch_size, expert_units, expert_num]
        """
        experts_output_list = []
        for i, experts in enumerate(self._experts_layer_list):
            output = input_tensor
            for j, expert_layer in enumerate(experts):
                output = expert_layer(output)
            experts_output_list.append(tf.expand_dims(output, [2]))
        return tf.concat(experts_output_list, axis=2)

    def get_experts_output(self):
        return self._experts_output

    @property
    def expert_num(self):
        return self._expert_num

    @property
    def expert_units(self):
        return self._expert_units


class Gates(object):
    def __init__(self, task_expert_list, selector_tensor, scope,
                 use_bias=True, initializer=tf.compat.v1.glorot_uniform_initializer,
                 regularizer=None, dtype=tf.float32):
        """Constructor for Gates
        Args:
            task_expert_list: A list of Experts Object.
            selector_tensor: Float Tensor of shape [batch_size, selector_size].
            scope: Expert scope.
            use_bias: Boolean, whether the layer uses a bias.
            initializer: Initializer function for the gates weight matrix.
            regularizer: Regularizer function for the gates weight matrix.
            reuse: Whether to reuse the variable.
        """
        input_tensor_list = []
        total_experts_num = 0
        for task_experts in task_expert_list:
            self._expert_units = task_experts.expert_units
            total_experts_num += task_experts.expert_num
            input_tensor_list.append(task_experts.get_experts_output())

        self._gates_layer = tf.keras.layers.Dense(units=total_experts_num,
                                                  activation=tf.nn.softmax,
                                                  name='%s/gates' % scope,
                                                  use_bias=use_bias,
                                                  kernel_initializer=initializer,
                                                  kernel_regularizer=regularizer,
                                                  dtype=dtype,
                                                  activity_regularizer=tf.keras.regularizers.l2(),
                                                  bias_regularizer=tf.keras.regularizers.l2
                                                  )

        self._gates_output = self._compute_gates_output(input_tensor_list, selector_tensor)

    def _compute_gates_weights(self, selector_tensor):
        """Compute gates weights
        Args:
            selector_tensor: Float Tensor of shape [batch_size, selector_size].
        Returns:
            Gate weights of shape [batch_size, total_experts_num].
            The "total_experts_num" is the sum of total input Experts number.
        """
        return tf.expand_dims(self._gates_layer(selector_tensor), axis=1)

    def _compute_gates_output(self, input_tensor_list, selector_tensor):
        """Compute gates output
        Args:
            input_tensor_list: A list of all Expert output tensor.
            selector_tensor: Float Tensor of shape [batch_size, selector_size].
        Returns:
            Gate output of shape [batch_size, expert_units].
        """
        gates_weights = self._compute_gates_weights(selector_tensor)
        input_tensor = tf.concat(input_tensor_list, axis=2)
        gates_output = tf.multiply(input_tensor, gates_weights)
        gates_output = tf.reduce_sum(gates_output, axis=2)
        gates_output = tf.reshape(gates_output, [-1, self._expert_units])
        return gates_output

    def get_gates_output(self):
        return self._gates_output


class PleNetworks(object):
    def __init__(self, input_tensor,
                 shared_expert_num,
                 task_expert_num_list,
                 level_expert_units_list,
                 task_name_list=None,
                 ):
        """Constructor for PleNetworks
        Args:
            input_tensor: Float Tensor of shape [batch_size, input_embedding_size].
            shared_expert_num: Int scalar, expert number of shared task.
            task_expert_num_list: A list of int for each task expert number.
            level_expert_units_list: A 2-D list, eg: [[256, 128], [128, 64]] means construct a 2-level ple networks
                               which the first level expert units is 256, 128 and the second level expert units is 128, 64.
            task_name_list: (optional)A list of string for each task name. The length of task_expert_num_list and task_name_list must be the same.
        """

        assert_rank(input_tensor, 2)

        task_num = len(task_expert_num_list)
        task_output_list = [input_tensor] * task_num
        shared_output = input_tensor

        if ((not isinstance(task_name_list, list)) or (len(task_name_list) != task_num)):
            task_name_list = [str(i) for i in range(task_num)]

        total_level_num = len(level_expert_units_list)
        with tf.compat.v1.variable_scope('ple'):
            for level, expert_units_list in enumerate(level_expert_units_list):
                with tf.compat.v1.variable_scope('level_%d' % level):
                    shared_experts = Experts(shared_output, shared_expert_num, 'shared', expert_units_list)

                    task_experts_list = []
                    for task_id, (task_expert_num, task_name) in enumerate(zip(task_expert_num_list, task_name_list)):
                        task_experts = Experts(task_output_list[task_id], task_expert_num, 'task_%s' % task_name,
                                               expert_units_list)
                        task_experts_list.append(task_experts)
                        task_output_list[task_id] = Gates([task_experts, shared_experts],
                                                          selector_tensor=task_output_list[task_id],
                                                          scope='task_%s' % task_name).get_gates_output()
                    if (level < total_level_num - 1):
                        shared_output = Gates(task_experts_list + [shared_experts],
                                              selector_tensor=shared_output,
                                              scope='shared').get_gates_output()

        # A list of tensor with shape: [batch_size, expert_units].
        self._task_output_list = task_output_list

    def get_task_output(self):
        return self._task_output_list
