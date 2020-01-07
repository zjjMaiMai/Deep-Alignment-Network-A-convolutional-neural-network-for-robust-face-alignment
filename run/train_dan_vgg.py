from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import cv2
import time
import argparse
from pathlib import Path
from easydict import EasyDict

from models.dan_vgg import DAN_VGG
from dataset.dataset_300w import Dataset300W, Container300W
from trainer.base_trainer import BaseTrainer
from trainer.base_model import LandmarkWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="fa")
    parser.add_argument('--model_dir', required=True, type=Path)
    parser.add_argument('--dataset_dir', required=True, type=Path)
    parser.add_argument('--cache_dir', required=True, type=Path)
    flags = parser.parse_args()
    return flags


def main():
    flags = parse_args()
    config = EasyDict(
        learning_rate=0.5,
        weight_decay=5e-4,
        momentum=0.9,
        num_steps=50000,
        batch_size=64,
        input_size=112
    )
    trainset = Dataset300W(
        Container300W([
            flags.dataset_dir / 'afw',
            flags.dataset_dir / 'helen/trainset',
            flags.dataset_dir / 'lfpw/trainset'],
            flags.cache_dir / 'train_cache',
            cache_size=config.input_size * 2
        ),
        output_size=config.input_size,
        augment=True)
    evalset = Dataset300W(
        Container300W([
            flags.dataset_dir / 'ibug',
            flags.dataset_dir / 'helen/testset',
            flags.dataset_dir / 'lfpw/testset'],
            flags.cache_dir / 'eval_cache',
            cache_size=config.input_size * 2
        ),
        output_size=config.input_size,
        augment=False)

    model = DAN_VGG(config.input_size, trainset.mean_shape, 0)

    model.stage = 0
    model_wrapper = LandmarkWrapper(model, config, trainset, evalset)
    BaseTrainer(model_dir=flags.model_dir / 'first_stage', num_step=config.num_steps,
                log_every_n_step=200).fit(model_wrapper)

    model.stage = 1
    model_wrapper = LandmarkWrapper(model, config, trainset, evalset)
    BaseTrainer(model_dir=flags.model_dir / 'second_stage', num_step=config.num_steps,
                log_every_n_step=200, fine_tune=True).fit(model_wrapper)


if __name__ == "__main__":
    from utils.misc.custom_module import random_seed
    random_seed(2019)
    main()
