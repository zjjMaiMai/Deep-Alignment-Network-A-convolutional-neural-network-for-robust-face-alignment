from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import cv2
import tensorflow as tf

from dan.to_tfrecord import CACHE_SIZE, CACHE_PADDING
from dan.predefine import get_300w_mean_shape, get_300w_flip_idx
from utils.transform.trans2d import from_translate, from_center_rotate, from_scale, fix_opencv_


def model_input_fn(path,
                   padding=0.3,
                   out_size=256,
                   data_augment=False):
    def _process(img_raw, lmk_raw):
        img = np.frombuffer(img_raw, dtype=np.uint8).reshape(
            CACHE_SIZE, CACHE_SIZE, 3)
        lmk = np.frombuffer(lmk_raw, dtype=np.float32).reshape(
            get_300w_mean_shape().shape)

        transform = from_scale(out_size / CACHE_SIZE)
        transform = from_center_rotate(
            (out_size / 2, out_size / 2), 0, (1 - padding) / (1 - CACHE_PADDING)) @ transform
        if data_augment:
            rs = from_center_rotate(
                (out_size / 2, out_size / 2),
                np.random.uniform(-10, 10),
                np.random.uniform(0.9, 1.1))
            t = from_translate(np.random.uniform(-0.1, 0.1, 2) * out_size)
            transform = t @ rs @ transform

        transform = transform.astype(np.float32)
        color = (np.random.randint(0, 256), np.random.randint(
            0, 256), np.random.randint(0, 256))
        img = cv2.warpAffine(img, fix_opencv_(transform)[:2, :], (out_size, out_size),
                             flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=color)
        lmk = lmk @ transform[:2, :2].T + transform[:2, 2]

        if data_augment and random.choices((True, False)):
            img = cv2.flip(img, 1)
            lmk = lmk[get_300w_flip_idx(), :]
            lmk[:, 0] = img.shape[1] - 1 - lmk[:, 0]

        img = img.astype(np.float32) / 255
        lmk = lmk / out_size
        return img, lmk

    feature_description = {
        'image_raw': tf.FixedLenFeature([], tf.string),
        'lmk_raw': tf.FixedLenFeature([], tf.string),
    }

    def _data_augment_tf_func(proto):
        dict_feature = tf.parse_single_example(proto, feature_description)
        img_tensor, lmk_tensor = tf.numpy_function(
            _process,
            inp=[dict_feature['image_raw'], dict_feature['lmk_raw']],
            Tout=(tf.float32, tf.float32))
        img_tensor.set_shape((out_size, out_size, 3))
        lmk_tensor.set_shape(get_300w_mean_shape().shape)
        return img_tensor, lmk_tensor

    files = tf.data.Dataset.list_files(os.path.join(path, "*.tfrecord"))
    dataset = files.interleave(
        lambda x: tf.data.TFRecordDataset(x).map(_data_augment_tf_func), cycle_length=32, num_parallel_calls=8)
    return dataset


if __name__ == "__main__":
    import argparse
    import pathlib

    parse = argparse.ArgumentParser()
    parse.add_argument("--path", type=str, required=True)
    flags = parse.parse_args()

    OUT_SIZE = 512
    dataset = model_input_fn(flags.path, out_size=OUT_SIZE).prefetch(1)
    next_element = dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        draw_shiftbits = 4
        draw_multiplier = 1 << 4

        while True:
            img, lmk = sess.run(next_element)

            for p in lmk:
                p_int = np.round(p * draw_multiplier *
                                 OUT_SIZE).astype(np.int32)
                cv2.circle(img, tuple(p_int), draw_multiplier * 2,
                           (255, 0, 0), -1, cv2.LINE_AA, draw_shiftbits)

            cv2.imshow("img", img)
            cv2.waitKey(-1)
