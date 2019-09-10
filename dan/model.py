from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from dan.predefine import get_300w_mean_shape
from utils.transform.trans2d import from_translate

L2_WEIGHT_DECAY = 1.0
INPUT_SIZE = 112

_STAGE_PADDING = 0.2
_STAGE_MEANSHAPE = ((get_300w_mean_shape() - 0.5) *
                    (1 - _STAGE_PADDING) + 0.5) * INPUT_SIZE
LMK_NUM = _STAGE_MEANSHAPE.shape[0]


def _conv(inputs, c, k, s, training, use_bn=True, use_relu=True):
    with tf.compat.v1.variable_scope('conv', reuse=False):
        x = tf.layers.conv2d(
            inputs=inputs,
            filters=c,
            kernel_size=k,
            strides=s,
            padding='same',
            activation=None,
            use_bias=(not use_bn),
            kernel_initializer=tf.compat.v1.initializers.he_uniform(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_WEIGHT_DECAY))
        if use_bn:
            x = tf.layers.batch_normalization(x, training=training)
        if use_relu:
            x = tf.nn.relu6(x)
        else:
            x = tf.identity(x)
    return x


def _depthwise_conv(inputs, k, s, training, use_bn=True, use_relu=True):
    with tf.compat.v1.variable_scope('dwconv'):
        inputs_shape = inputs.get_shape()
        input_c = inputs_shape[-1].value
        kernel = tf.compat.v1.get_variable(
            name="dwconv_conv_kernel",
            shape=(k, k, input_c, 1),
            initializer=tf.compat.v1.initializers.he_uniform(),
            regularizer=tf.contrib.layers.l2_regularizer(L2_WEIGHT_DECAY),
            trainable=True)

        x = tf.nn.depthwise_conv2d(
            input=inputs,
            filter=kernel,
            strides=(1, s, s, 1),
            padding='SAME')
        if use_bn:
            x = tf.layers.batch_normalization(x, training=training)
        if use_relu:
            x = tf.nn.relu6(x)
        else:
            x = tf.identity(x)
    return x


def _one_block_s1(inputs, c_in, c_out, training, s=1, expand=2):
    x = inputs
    if c_in <= c_out:
        with tf.compat.v1.variable_scope('proj'):
            x = _conv(x, c=c_in * expand, k=1, s=1, training=training)
    with tf.compat.v1.variable_scope('dw_pw'):
        x = _depthwise_conv(x, k=3, s=s, training=training)
        x = _conv(x, c=c_out, k=1, s=1, use_relu=False, training=training)
    if s == 1 and c_in == c_out:
        x = inputs + x
    return x


def _umeyama_tf(src, dst):
    src_mean = tf.reduce_mean(src, axis=1, keep_dims=True)
    dst_mean = tf.reduce_mean(dst, axis=1, keep_dims=True)

    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    dot_result = tf.einsum('ijk,ijk->i', src_demean, dst_demean)
    norm_pow_2 = tf.pow(tf.norm(src_demean, axis=[1, 2]), 2)

    a = dot_result / norm_pow_2
    b = tf.reduce_sum(tf.multiply(src_demean[:, :, 0], dst_demean[:, :, 1]) - tf.multiply(
        src_demean[:, :, 1], dst_demean[:, :, 0]), axis=1) / norm_pow_2

    sr = tf.reshape(tf.stack([a, -b, b, a], axis=1), [-1, 2, 2])
    t = dst_mean - tf.matmul(src_mean, tf.linalg.matrix_transpose(sr))

    sr = tf.pad(sr, [[0, 0], [0, 1], [0, 0]])
    t = tf.linalg.matrix_transpose(
        tf.pad(t, [[0, 0], [0, 0], [0, 1]], constant_values=1))

    ret = tf.concat([sr, t], axis=-1)
    return ret


def _transfrom_lmk(lmk, transform, inv=False):
    if inv:
        transform = tf.linalg.inv(transform)
    lmk = tf.matmul(lmk, transform[:, :2, :2],
                    transpose_b=True) + tf.linalg.matrix_transpose(transform[:, :2, 2:])
    return lmk


def _transfrom_img(img, transform, inv=False):
    if not inv:
        transform = tf.linalg.inv(transform)
    # fix shift 0.5 pix
    transform = tf.linalg.matmul(
        transform, from_translate(-0.5).astype(np.float32))
    transform = tf.linalg.matmul(
        from_translate(0.5).astype(np.float32), transform)
    transform = tf.contrib.image.matrices_to_flat_transforms(transform)
    img = tf.contrib.image.transform(img, transform, interpolation='NEAREST')
    return img


def _gen_heatmap(shapes):
    pixels = tf.constant([(x, y) for y in range(INPUT_SIZE) for x in range(
        INPUT_SIZE)], dtype=tf.float32, shape=[1, INPUT_SIZE, INPUT_SIZE, 2])
    shapes = shapes[:, :, tf.newaxis, tf.newaxis, :]
    value = pixels - shapes
    value = tf.norm(value, axis=-1)
    value = 1.0 / (tf.reduce_min(value, axis=1) + 1.0)
    value = tf.expand_dims(value, axis=-1)
    return value


def _first_stage_model_fn(inputs, training=True):
    with tf.compat.v1.variable_scope("stage_0"):
        with tf.compat.v1.variable_scope('head'):
            x = _conv(inputs, c=24, k=3, s=2, training=training)

        with tf.compat.v1.variable_scope('backbone'):
            last_channel = 24
            layers_config = [
                # c, n, s, e
                [16, 3, 2, 4],
                [32, 3, 2, 4],
                [64, 4, 2, 4],
                [96, 4, 1, 4],
            ]
            for idx, _value in enumerate(layers_config):
                c, n, s, e = _value
                with tf.compat.v1.variable_scope('layer_{}'.format(idx)):
                    for count in range(n):
                        with tf.compat.v1.variable_scope('block_{}'.format(count)):
                            x = _one_block_s1(x, last_channel, c, s=(
                                s if count == 0 else 1), expand=e, training=training)
                            last_channel = c

        with tf.compat.v1.variable_scope('gap'):
            x = _conv(x, 256, 1, 1, training=training)
            feature = tf.layers.average_pooling2d(x, 7, 7)
        with tf.compat.v1.variable_scope('fc'):
            x = _conv(feature, c=LMK_NUM * 2, k=1, s=1, training=training,
                      use_bn=False, use_relu=False)
        x = tf.reshape(x, shape=[-1, LMK_NUM, 2], name="lmk")
    return x, feature


def _other_stage_model_fn(inputs, last_stage_lmk, last_stage_feature, stage_idx, training=True):
    with tf.compat.v1.variable_scope("stage_{}".format(stage_idx)):
        with tf.compat.v1.variable_scope('affine'):
            transform = _umeyama_tf(
                last_stage_lmk, _STAGE_MEANSHAPE[np.newaxis, ::])
            inputs = _transfrom_img(inputs, transform, inv=False)
            transform_lmk = _transfrom_lmk(
                last_stage_lmk, transform, inv=False)
            heatmap = _gen_heatmap(transform_lmk)
            last_stage_feature = _conv(
                last_stage_feature, c=56 ** 2, k=1, s=1, training=training)
            last_stage_feature = tf.reshape(
                last_stage_feature, [-1, 56, 56, 1])
            last_stage_feature = tf.compat.v1.image.resize_nearest_neighbor(
                last_stage_feature, [112, 112], align_corners=True)
            last_stage_feature = _transfrom_img(
                last_stage_feature, transform, inv=False)

        inputs = tf.concat([inputs, heatmap, last_stage_feature], axis=-1)
        with tf.compat.v1.variable_scope('head'):
            x = _conv(inputs, c=24, k=3, s=2, training=training)

        with tf.compat.v1.variable_scope('backbone'):
            last_channel = 24
            layers_config = [
                # c, n, s, e
                [16, 3, 2, 4],
                [32, 3, 2, 4],
                [64, 4, 2, 4],
                [96, 4, 1, 4],
            ]
            for idx, _value in enumerate(layers_config):
                c, n, s, e = _value
                with tf.compat.v1.variable_scope('layer_{}'.format(idx)):
                    for count in range(n):
                        with tf.compat.v1.variable_scope('block_{}'.format(count)):
                            x = _one_block_s1(x, last_channel, c, s=(
                                s if count == 0 else 1), expand=e, training=training)
                            last_channel = c

        with tf.compat.v1.variable_scope('gap'):
            x = _conv(x, 256, 1, 1, training=training)
            feature = tf.layers.average_pooling2d(x, 7, 7)
        with tf.compat.v1.variable_scope('fc'):
            x = _conv(feature, c=LMK_NUM * 2, k=1, s=1, training=training,
                      use_bn=False, use_relu=False)
        x = tf.reshape(x, shape=[-1, LMK_NUM, 2])
        x = x + transform_lmk
        x = _transfrom_lmk(x, transform, inv=True)
        return x, feature


def model_fn(inputs, train_stage_idx):
    s0_lmk, feature = _first_stage_model_fn(
        inputs, training=(train_stage_idx == 0))
    s1_lmk, feature = _other_stage_model_fn(
        inputs, s0_lmk, feature, 1, training=(train_stage_idx == 1))
    return s0_lmk, s1_lmk


if __name__ == '__main__':
    _, output = model_fn(tf.compat.v1.placeholder(
        tf.float32, shape=[1, 112, 112, 3]), -1)
    tf.compat.v1.profiler.profile(
        tf.get_default_graph(),
        cmd='op',
        options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    print("Output Tensor : {}".format(output))
