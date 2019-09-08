from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from dan.predefine import get_300w_mean_shape

L2_WEIGHT_DECAY = 1.0
LMK_NUM = get_300w_mean_shape().shape[0]


def _conv(inputs, c, k, s, training, use_bn=True, use_relu=True):
    with tf.variable_scope('conv', reuse=False):
        x = tf.layers.conv2d(
            inputs=inputs,
            filters=c,
            kernel_size=k,
            strides=s,
            padding='same',
            activation=None,
            use_bias=(not use_bn),
            kernel_initializer=tf.initializers.he_uniform(),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_WEIGHT_DECAY))
        if use_bn:
            x = tf.layers.batch_normalization(x, training=training)
        if use_relu:
            x = tf.nn.relu6(x)
        else:
            x = tf.identity(x)
    return x


def _depthwise_conv(inputs, k, s, training, use_bn=True, use_relu=True):
    with tf.variable_scope('dwconv'):
        inputs_shape = inputs.get_shape()
        input_c = inputs_shape[-1].value
        kernel = tf.get_variable(
            name="dwconv_conv_kernel",
            shape=(k, k, input_c, 1),
            initializer=tf.initializers.he_uniform(),
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
        with tf.variable_scope('proj'):
            x = _conv(x, c=c_in * expand, k=1, s=1, training=training)
    with tf.variable_scope('dw_pw'):
        x = _depthwise_conv(x, k=3, s=s, training=training)
        x = _conv(x, c=c_out, k=1, s=1, use_relu=False, training=training)
    if s == 1 and c_in == c_out:
        x = inputs + x
    return x


def _umeyama_tf(src_lmk, dst_lmk):
    return None


def _transfrom_img(img, transform, inv=False):
    return None


def _transfrom_lmk(lmk, transform, inv=False):
    return None


def _gen_heatmap(lmk):
    return None


def _first_stage_model_fn(inputs, training=True):
    with tf.variable_scope("stage_0"):
        with tf.variable_scope('head'):
            x = _conv(inputs, c=24, k=3, s=2, training=training)

        with tf.variable_scope('backbone'):
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
                with tf.variable_scope('layer_{}'.format(idx)):
                    for count in range(n):
                        with tf.variable_scope('block_{}'.format(count)):
                            x = _one_block_s1(x, last_channel, c, s=(
                                s if count == 0 else 1), expand=e, training=training)
                            last_channel = c

        with tf.variable_scope('gap_fc'):
            x = _conv(x, 256, 1, 1, training=training)
            feature = tf.layers.average_pooling2d(x, 7, 7)
            x = _conv(feature, c=LMK_NUM * 2, k=1, s=1, training=training,
                      use_bn=False, use_relu=False)
        x = tf.reshape(x, shape=[-1, LMK_NUM, 2], name="lmk")
    return x, feature


def _other_stage_model_fn(inputs, last_stage_lmk, last_stage_feature, stage_idx, training=True):
    with tf.variable_scope("stage_{}".format(stage_idx)):
        with tf.variable_scope('head'):
            transform = _umeyama_tf(last_stage_lmk)
            inputs = _transfrom_img(inputs, transform, inv=False)
            transform_lmk = _transfrom_lmk(
                last_stage_lmk, transform, inv=False)
            heatmap = _gen_heatmap(transform_lmk)
            last_stage_feature = _conv(
                last_stage_feature, c=56 ** 2, k=1, s=1, training=training)
            last_stage_feature = tf.reshape(
                last_stage_feature, [-1, 56, 56, 1])
            last_stage_feature = tf.image.resize_nearest_neighbor(
                last_stage_feature, [112, 112], align_corners=True)

            inputs = tf.concat([inputs, heatmap, last_stage_feature], axis=-1)
            x = _conv(inputs, c=24, k=3, s=2, training=training)

        with tf.variable_scope('backbone'):
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
                with tf.variable_scope('layer_{}'.format(idx)):
                    for count in range(n):
                        with tf.variable_scope('block_{}'.format(count)):
                            x = _one_block_s1(x, last_channel, c, s=(
                                s if count == 0 else 1), expand=e, training=training)
                            last_channel = c

        with tf.variable_scope('gap_fc'):
            x = _conv(x, 256, 1, 1, training=training)
            feature = tf.layers.average_pooling2d(x, 7, 7)
            x = _conv(feature, c=LMK_NUM * 2, k=1, s=1, training=training,
                      use_bn=False, use_relu=False)
        x = tf.reshape(x, shape=[-1, LMK_NUM, 2], name="lmk")
        return x, feature


def model_fn(inputs, training=True):
    inputs = tf.identity(inputs, name="image")
    lmk, feature = _first_stage_model_fn(inputs, training=training)
    lmk, feature = _other_stage_model_fn(
        inputs, lmk, feature, 1, training=training)
    lmk = tf.identity(lmk, "lmk")
    return lmk


if __name__ == '__main__':
    output = model_fn(tf.placeholder(
        tf.float32, shape=[1, 112, 112, 3]), False)
    tf.profiler.profile(
        tf.get_default_graph(),
        cmd='op',
        options=tf.profiler.ProfileOptionBuilder.float_operation())
    print("Output Tensor : {}".format(output))


# class Model(object):

#     def __init__(self,
#                  num_lmark,
#                  img_size,
#                  filter_sizes,
#                  num_convs,
#                  kernel_size,
#                  data_format=None):
#         if not data_format:
#             data_format = (
#                 'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

#         self.data_format = data_format
#         self.filter_sizes = filter_sizes
#         self.num_convs = num_convs
#         self.num_lmark = num_lmark
#         self.kernel_size = kernel_size
#         self.img_size = img_size

#         self.__pixels__ = tf.constant([(x, y) for y in range(self.img_size) for x in range(self.img_size)],
#                                       dtype=tf.float32, shape=[1, self.img_size, self.img_size, 2])
#         # self.__pixels__ = tf.tile(self.__pixels__,[num_lmark,1,1,1])

#     def __calc_affine_params(self, from_shape, to_shape):
#         from_shape = tf.reshape(from_shape, [-1, self.num_lmark, 2])
#         to_shape = tf.reshape(to_shape, [-1, self.num_lmark, 2])

#         from_mean = tf.reduce_mean(from_shape, axis=1, keepdims=True)
#         to_mean = tf.reduce_mean(to_shape, axis=1, keepdims=True)

#         from_centralized = from_shape - from_mean
#         to_centralized = to_shape - to_mean

#         dot_result = tf.reduce_sum(tf.multiply(from_centralized, to_centralized),
#                                    axis=[1, 2])
#         norm_pow_2 = tf.pow(tf.norm(from_centralized, axis=[1, 2]), 2)

#         a = dot_result / norm_pow_2
#         b = tf.reduce_sum(tf.multiply(from_centralized[:, :, 0], to_centralized[:, :, 1]) - tf.multiply(
#             from_centralized[:, :, 1], to_centralized[:, :, 0]), 1) / norm_pow_2

#         r = tf.reshape(tf.stack([a, b, -b, a], axis=1), [-1, 2, 2])
#         t = to_mean - tf.matmul(from_mean, r)
#         return r, t

#     def __affine_image(self, imgs, r, t):
#         # The Tensor [imgs].format is [NHWC]
#         r = tf.matrix_inverse(r)
#         r = tf.matrix_transpose(r)

#         rm = tf.reshape(
#             tf.pad(r, [[0, 0], [0, 0], [0, 1]], mode='CONSTANT'), [-1, 6])
#         rm = tf.pad(rm, [[0, 0], [0, 2]], mode='CONSTANT')

#         tm = tf.contrib.image.translations_to_projective_transforms(
#             tf.reshape(t, [-1, 2]))
#         rtm = tf.contrib.image.compose_transforms(rm, tm)

#         return tf.contrib.image.transform(imgs, rtm, "BILINEAR")

#     def __affine_shape(self, shapes, r, t, isinv=False):
#         if isinv:
#             r = tf.matrix_inverse(r)
#             t = tf.matmul(-t, r)
#         shapes = tf.matmul(shapes, r) + t
#         return shapes

#     def __gen_heatmap(self, shapes):
#         shapes = shapes[:, :, tf.newaxis, tf.newaxis, :]
#         value = self.__pixels__ - shapes
#         value = tf.norm(value, axis=-1)
#         value = 1.0 / (tf.reduce_min(value, axis=1) + 1.0)
#         value = tf.expand_dims(value, axis=-1)
#         return value

#     def __call__(self,
#                  inputs_imgs,
#                  s1_training,
#                  s2_training,
#                  mean_shape,
#                  imgs_mean,
#                  imgs_std):
#         rd = {}
#         inputs_imgs = tf.reshape(
#             inputs_imgs, [-1, self.img_size, self.img_size, 1])
#         tf.summary.image('image', inputs_imgs, max_outputs=6)

#         rd['img'] = inputs_imgs

#         mean_shape = tf.reshape(mean_shape, [self.num_lmark, 2]) if mean_shape is not None else tf.zeros(
#             [self.num_lmark, 2], tf.float32)
#         imgs_mean = tf.reshape(imgs_mean, [self.img_size, self.img_size, 1]) if imgs_mean is not None else tf.zeros(
#             [self.img_size, self.img_size, 1], tf.float32)
#         imgs_std = tf.reshape(imgs_std, [self.img_size, self.img_size, 1]) if imgs_std is not None else tf.ones(
#             [self.img_size, self.img_size, 1], tf.float32)

#         imgs_mean_tensor = tf.get_variable(
#             'imgs_mean', trainable=False, initializer=imgs_mean)
#         imgs_std_tensor = tf.get_variable(
#             'imgs_std', trainable=False, initializer=imgs_std)
#         shape_mean_tensor = tf.get_variable(
#             'shape_mean', trainable=False, initializer=mean_shape)

#         inputs_imgs = (inputs_imgs - imgs_mean_tensor) / imgs_std_tensor
#         # Convert the inputs from channels_last (NHWC) to channels_first
#         # (NCHW).
#         # This provides a large performance boost on GPU.  See
#         # https://www.tensorflow.org/performance/performance_guide#data_formats
#         with tf.variable_scope('s1'):
#             inputs = inputs_imgs

#             if self.data_format == 'channels_first':
#                 inputs = tf.transpose(inputs, [0, 3, 1, 2])

#             for i, num_filter in enumerate(self.filter_sizes):
#                 inputs = vgg_block(inputs=inputs, filters=num_filter, num_convs=self.num_convs,
#                                    training=s1_training, kernel_size=self.kernel_size, maxpool=True,
#                                    data_format=self.data_format)

#             inputs = tf.contrib.layers.flatten(inputs)
#             inputs = tf.layers.dropout(inputs, 0.5, training=s1_training)

#             s1_fc1 = tf.layers.dense(
#                 inputs, 256, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())
#             s1_fc1 = batch_norm(s1_fc1, s1_training,
#                                 data_format=self.data_format)

#             s1_fc2 = tf.layers.dense(
#                 s1_fc1, self.num_lmark * 2, activation=None)
#             rd['s1_ret'] = tf.identity(tf.reshape(
#                 s1_fc2, [-1, self.num_lmark, 2]) + shape_mean_tensor, name='output_landmark')

#         with tf.variable_scope('s2'):
#             r, t = self.__calc_affine_params(rd['s1_ret'], shape_mean_tensor)
#             inputs = self.__affine_image(inputs_imgs, r, t)
#             s2_lmark = self.__affine_shape(rd['s1_ret'], r, t)
#             s2_heatmap = self.__gen_heatmap(s2_lmark)
#             s2_feature = tf.layers.dense(
#                 s1_fc1, (self.img_size // 2) ** 2, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())

#             s2_feature = tf.reshape(
#                 s2_feature, [-1, self.img_size // 2, self.img_size // 2, 1])
#             s2_feature_upscale = tf.image.resize_images(
#                 s2_feature, [self.img_size, self.img_size])

#             tf.summary.image('heatmap', s2_heatmap, max_outputs=6)
#             tf.summary.image('feature', s2_feature, max_outputs=6)
#             tf.summary.image('image', inputs, max_outputs=6)

#             if self.data_format == 'channels_first':
#                 inputs = tf.transpose(inputs, [0, 3, 1, 2])
#                 s2_heatmap = tf.transpose(s2_heatmap, [0, 3, 1, 2])
#                 s2_feature_upscale = tf.transpose(
#                     s2_feature_upscale, [0, 3, 1, 2])

#             inputs = tf.concat([inputs, s2_heatmap, s2_feature_upscale],
#                                axis=1 if self.data_format == 'channels_first' else 3)
#             inputs = batch_norm(inputs, s2_training, self.data_format)

#             for i, num_filter in enumerate(self.filter_sizes):
#                 inputs = vgg_block(inputs=inputs, filters=num_filter, num_convs=self.num_convs,
#                                    training=s2_training, kernel_size=self.kernel_size, maxpool=True,
#                                    data_format=self.data_format)

#             inputs = tf.contrib.layers.flatten(inputs)
#             inputs = tf.layers.dropout(inputs, 0.5, training=s2_training)

#             s2_fc1 = tf.layers.dense(
#                 inputs, 256, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer())
#             s2_fc1 = batch_norm(s2_fc1, s2_training,
#                                 data_format=self.data_format)

#             s2_fc2 = tf.layers.dense(
#                 s2_fc1, self.num_lmark * 2, activation=None)
#             s2_fc2 = tf.reshape(s2_fc2, [-1, self.num_lmark, 2]) + s2_lmark
#             rd['s2_ret'] = tf.identity(self.__affine_shape(
#                 s2_fc2, r, t, isinv=True), name='output_landmark')

#         return rd
