from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pathlib
import argparse
import numpy as np
import tensorflow as tf

from utils.custom_module.tf import *
from dan.model import model_fn, INPUT_SIZE
from dan.dataset import model_input_fn

PADDING = 0.3
LR = 0.2
WEIGHT_DECAY = 5e-5
BATCH_SIZE = 128
NUM_STEPS = 100000

def parse_args():
    parser = argparse.ArgumentParser(description="dan")

    parser.add_argument('--model_dir', required=True, type=str)
    parser.add_argument('--trainset_dir', required=True, type=str)
    parser.add_argument('--evalset_dir', type=str)
    flags = parser.parse_args()
    return flags


def build_model(image, label, flags):
    image = tf.identity(image, name="image")
    lmk = model_fn(image, train_stage_idx=0)
    lmk = tf.identity(lmk, name="landmark")

    fa_loss = tf.reduce_mean(ibug_score(label, lmk))

    regularization_loss = tf.losses.get_regularization_loss()
    total_loss = fa_loss + regularization_loss * WEIGHT_DECAY

    learning_rate = cosine_decay_with_warmup(
        tf.train.get_or_create_global_step(),
        LR,
        NUM_STEPS,
        LR * 0.1,
        warmup_steps=NUM_STEPS // 10)
    train_op = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=0.9).minimize(
        total_loss, tf.train.get_or_create_global_step())
    update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group([train_op, update_ops])

    tf.summary.scalar('fa_loss', fa_loss)
    tf.summary.scalar('regularization_loss', regularization_loss)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('learning_rate', learning_rate)
    return train_op


def main():
    flags = parse_args()
    pathlib.Path(flags.model_dir).mkdir(parents=True, exist_ok=True)
    print(flags)

    train_dataset = model_input_fn(
        flags.trainset_dir,
        padding=PADDING,
        out_size=INPUT_SIZE,
        data_augment=True).shuffle(2000).repeat(20).batch(BATCH_SIZE, drop_remainder=True).prefetch(1)
    iterator = train_dataset.make_initializable_iterator()
    next_img, next_label = iterator.get_next()
    train_op = build_model(next_img, next_label, flags)

    sess = tf.Session()
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(
        flags.model_dir + '/summary/train', sess.graph)
    sess.run([tf.global_variables_initializer(), iterator.initializer])
    while True:
        step = sess.run(tf.train.get_or_create_global_step())
        if step >= NUM_STEPS:
            break
        try:
            summary, _ = sess.run([merged, train_op])
            train_writer.add_summary(summary, step)
        except tf.errors.OutOfRangeError:
            sess.run(iterator.initializer)
    saver.save(sess, flags.model_dir + '/train_saver/model', global_step=step)


if __name__ == "__main__":
    main()
