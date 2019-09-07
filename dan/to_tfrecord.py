from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pathlib
import numpy as np
import cv2
import tensorflow as tf

from dan.predefine import get_300w_mean_shape
from utils.transform.umeyama import umeyama

_output_size = 256
_crop_mean = ((get_300w_mean_shape() - 0.5) * 0.6 + 0.5) * _output_size


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--input_dir", nargs='+', required=True, type=str)
    parse.add_argument("--output_dir", required=True, type=str)
    flags = parse.parse_args()
    return flags


def sample_to_example(img_path):
    img_path = img_path.decode()
    pts_path = str(pathlib.Path(img_path).with_suffix('.pts'))

    img = cv2.imread(img_path)
    assert img is not None, "Can not read {}".format(img_path)

    lmk = np.genfromtxt(pts_path, skip_header=3, skip_footer=1)
    lmk = lmk.reshape(-1, 2).astype(np.float32)

    transform = umeyama(lmk, _crop_mean)
    image = cv2.warpAffine(img, transform[:2, :], (_output_size, _output_size),
                           flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    lmk = lmk @ transform[:2, :2].T + transform[:2, 2]

    feature = {
        'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
        'lmk_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lmk.tobytes()]))
    }
    example = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def to_record(img_path_list, output_path):
    def map_func(input):
        example = tf.py_func(
            func=sample_to_example,
            inp=[input],
            Tout=tf.string)
        return example

    dataset = tf.data.Dataset.from_tensor_slices(img_path_list)
    print(dataset)
    dataset = dataset.map(
        map_func, num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(1)
    writer_op = tf.data.experimental.TFRecordWriter(output_path).write(dataset)

    with tf.Session() as sess:
        sess.run(writer_op)
    return


def main():
    flags = parse_args()
    output_dir = pathlib.Path(flags.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, d in enumerate(flags.input_dir):
        d = pathlib.Path(d)
        img_list = list(d.rglob('*.png')) + list(d.rglob('*.jpg'))
        img_list = list(map(str, img_list))
        to_record(img_list, str(output_dir / '{}.tfrecord'.format(idx)))


if __name__ == "__main__":
    main()
