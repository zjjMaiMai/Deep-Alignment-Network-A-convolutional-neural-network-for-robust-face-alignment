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
    parser.add_argument('--train_stage', default=0, type=int)
    flags = parser.parse_args()
    return flags


def build_model(features, labels, mode, params):
    image = tf.identity(features, name="image")
    s0_lmk, s1_lmk = model_fn(
        image, train_stage_idx=params['stage'] if mode == tf.estimator.ModeKeys.TRAIN else -1)
    s1_lmk = tf.identity(s1_lmk, name="landmark")

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'lmk': s1_lmk,
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.PREDICT,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    stage_id = 'stage_0' if params['stage'] == 0 else 'stage_1'
    lmk_out = s0_lmk if params['stage'] == 0 else s1_lmk
    fa_loss = tf.reduce_mean(ibug_score(labels, lmk_out))

    if mode == tf.estimator.ModeKeys.TRAIN:
        regularization_loss = tf.losses.get_regularization_loss(stage_id)
        total_loss = fa_loss + regularization_loss * WEIGHT_DECAY

        learning_rate = cosine_decay_with_warmup(
            tf.train.get_or_create_global_step(),
            LR,
            NUM_STEPS,
            LR * 0.1,
            warmup_steps=NUM_STEPS // 10)

        train_op = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=0.9).minimize(
            total_loss, tf.train.get_or_create_global_step(),
            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, stage_id))
        update_ops = tf.compat.v1.get_collection(
            tf.GraphKeys.UPDATE_OPS, stage_id)
        train_op = tf.group([train_op, update_ops])

        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=fa_loss,
            train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL,
            loss=fa_loss)


def main():
    flags = parse_args()
    pathlib.Path(flags.model_dir).mkdir(parents=True, exist_ok=True)
    print(flags)

    def train_input_fn():
        dataset = model_input_fn(
            flags.trainset_dir,
            padding=PADDING,
            out_size=INPUT_SIZE,
            data_augment=True).shuffle(2000).repeat().batch(BATCH_SIZE, drop_remainder=True).prefetch(1)
        return dataset

    def eval_input_fn():
        dataset = model_input_fn(
            flags.evalset_dir,
            padding=PADDING,
            out_size=INPUT_SIZE).batch(BATCH_SIZE).prefetch(1)
        return dataset

    estimator = tf.estimator.Estimator(
        model_fn=build_model, model_dir=flags.model_dir,
        params={
            'stage': flags.train_stage,
        })
    for _ in range(NUM_STEPS // 500):
        print('Starting a training cycle.')
        estimator.train(input_fn=train_input_fn, steps=500)

        print('Starting to evaluate.')
        eval_results = estimator.evaluate(input_fn=eval_input_fn)
        print(eval_results)


if __name__ == "__main__":
    main()
