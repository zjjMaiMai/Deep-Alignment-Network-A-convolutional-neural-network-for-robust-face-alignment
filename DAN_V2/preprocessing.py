from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os 
import sys
import glob
import random
import numpy as np
import cv2
import uuid

import tensorflow as tf

tf.app.flags.DEFINE_string('input_dir', None, "input_dir")
tf.app.flags.DEFINE_string('output_dir', None, "output_dir")
tf.app.flags.DEFINE_boolean('istrain', False, "istrain")
tf.app.flags.DEFINE_integer('repeat', 1, 'repeat')
tf.app.flags.DEFINE_integer('img_size', 112, 'img_size')
tf.app.flags.DEFINE_string('mirror_file', None, 'mirror_file')

FLAGS = tf.app.flags.FLAGS
BATCH_SIZE = 128

def getAffine(From, To):
    FromMean = np.mean(From, axis=0)
    ToMean = np.mean(To, axis=0)

    FromCentralized = From - FromMean
    ToCentralized = To - ToMean

    FromVector = (FromCentralized).flatten()
    ToVector = (ToCentralized).flatten()

    DotResult = np.dot(FromVector, ToVector)
    NormPow2 = np.linalg.norm(FromCentralized) ** 2

    a = DotResult / NormPow2
    b = np.sum(np.cross(FromCentralized, ToCentralized)) / NormPow2

    R = np.array([[a, b], [-b, a]])
    T = ToMean - np.dot(FromMean, R)

    return R, T


def _load_data(imagepath, ptspath, is_train,mirror_array):
    def makerotate(angle):
        rad = angle * np.pi / 180.0
        return np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]], dtype=np.float32)

    srcpts = np.genfromtxt(ptspath.decode(), skip_header=3, skip_footer=1)
    x, y = np.min(srcpts, axis=0).astype(np.int32)
    w, h = np.ptp(srcpts, axis=0).astype(np.int32)
    pts = (srcpts - [x, y]) / [w, h]

    img = cv2.imread(imagepath.decode(), cv2.IMREAD_GRAYSCALE)
    center = [0.5, 0.5]

    if is_train:
        pts = pts - center
        pts = np.dot(pts, makerotate(np.random.normal(0, 20)))
        pts = pts * np.random.normal(0.8, 0.05)
        pts = pts + [np.random.normal(0, 0.05),
                     np.random.normal(0, 0.05)] + center

        pts = pts * FLAGS.img_size

        R, T = getAffine(srcpts, pts)
        M = np.zeros((2, 3), dtype=np.float32)
        M[0:2, 0:2] = R.T
        M[:, 2] = T
        img = cv2.warpAffine(img, M, (FLAGS.img_size, FLAGS.img_size))

        if any(mirror_array) and random.choice((True, False)):
            pts[:,0] = FLAGS.img_size - 1 - pts[:,0]
            pts = pts[mirror_array]
            img = cv2.flip(img, 1)

    else:
        pts = pts - center
        pts = pts * 0.8
        pts = pts + center

        pts = pts * FLAGS.img_size

        R, T = getAffine(srcpts, pts)
        M = np.zeros((2, 3), dtype=np.float32)
        M[0:2, 0:2] = R.T
        M[:, 2] = T
        img = cv2.warpAffine(img, M, (FLAGS.img_size, FLAGS.img_size))


    _,filename = os.path.split(imagepath.decode())
    filename,_ = os.path.splitext(filename)

    uid = str(uuid.uuid1())

    cv2.imwrite(os.path.join(FLAGS.output_dir,filename + '@' + uid + '.png'),img)
    np.savetxt(os.path.join(FLAGS.output_dir,filename + '@' + uid + '.ptv'),pts,delimiter=',')

    return img,pts.astype(np.float32)

def _input_fn(img, pts, is_train,mirror_array):
    dataset_image = tf.data.Dataset.from_tensor_slices(img)
    dataset_pts = tf.data.Dataset.from_tensor_slices(pts)
    dataset = tf.data.Dataset.zip((dataset_image, dataset_pts))

    dataset = dataset.prefetch(BATCH_SIZE)
    dataset = dataset.repeat(FLAGS.repeat)
    dataset = dataset.map(lambda imagepath, ptspath: tuple(tf.py_func(_load_data, [
                          imagepath, ptspath, is_train,mirror_array], [tf.uint8,tf.float32])), num_parallel_calls=8)                     
    dataset = dataset.prefetch(1)

    return dataset

def _get_filenames(data_dir, listext):
    imagelist = []
    for ext in listext:
        p = os.path.join(data_dir, ext)
        imagelist.extend(glob.glob(p))

    ptslist = []
    for image in imagelist:
        ptslist.append(os.path.splitext(image)[0] + ".pts")

    return imagelist, ptslist

def main(argv):
    imagenames, ptsnames = _get_filenames(FLAGS.input_dir, ["*.jpg", "*.png"])
    mirror_array = np.genfromtxt(FLAGS.mirror_file, dtype=int, delimiter=',') if FLAGS.mirror_file else np.zeros(1)
    
    dataset = _input_fn(imagenames,ptsnames,FLAGS.istrain,mirror_array)
    next_element = dataset.make_one_shot_iterator().get_next()

    img_list = []
    pts_list = []

    with tf.Session() as sess:
        count = 0
        while True:
            try:
                img,pts = sess.run(next_element)
                img_list.append(img)
                pts_list.append(pts)
            except tf.errors.OutOfRangeError:
                img_list = np.stack(img_list)
                pts_list = np.stack(pts_list)

                mean_shape = np.mean(pts_list,axis=0)
                imgs_mean = np.mean(img_list,axis=0)
                imgs_std = np.std(img_list,axis=0)

                np.savetxt(os.path.join(FLAGS.output_dir,'mean_shape.ptv'),mean_shape,delimiter=',')
                np.savetxt(os.path.join(FLAGS.output_dir,'imgs_mean.ptv'),imgs_mean,delimiter=',')
                np.savetxt(os.path.join(FLAGS.output_dir,'imgs_std.ptv'),imgs_std,delimiter=',')

                print("end")
                break


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(argv=sys.argv)