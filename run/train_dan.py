from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import cv2
import time
import argparse
from pathlib import Path
from easydict import EasyDict

from models.model import DAN
from dataset.dataset_300w import Dataset300W
from trainer.base_trainer import BaseTrainer
from trainer.base_model import LandmarkWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="fa")
    parser.add_argument('--model_dir', required=True, type=Path)
    parser.add_argument('--trainset', nargs='+', required=True, type=Path)
    parser.add_argument('--evalset', nargs='+', required=True, type=Path)
    flags = parser.parse_args()
    return flags


def main():
    flags = parse_args()
    config = EasyDict(
        learning_rate=0.5,
        weight_decay=5e-4,
        momentum=0.9,
        num_steps=10000,
        batch_size=64,
    )
    trainset = Dataset300W(flags.trainset, augment=True)
    evalset = Dataset300W(flags.evalset, augment=False)

    model = DAN(trainset.mean_shape, 0)
    model_wrapper = LandmarkWrapper(model, config, trainset, evalset)
    BaseTrainer(model_dir=flags.model_dir / 'first_stage', num_step=config.num_steps,
                log_every_n_step=1).fit(model_wrapper)


if __name__ == "__main__":
    from utils.misc.custom_module import random_seed
    random_seed(2019)
    main()
