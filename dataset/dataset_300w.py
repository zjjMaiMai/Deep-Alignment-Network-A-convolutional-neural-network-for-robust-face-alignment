from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from dataset.base_dataset import *
from utils.transform.trans2d import *
from utils.transform.umeyama import *


MEANSHAPE_300W = np.array([
    [0., 0.19162136],
    [0.00342039, 0.32351251],
    [0.0180597, 0.45481963],
    [0.04585256, 0.5833164],
    [0.09756264, 0.70239563],
    [0.17664164, 0.80549044],
    [0.27326557, 0.89158001],
    [0.37968491, 0.96054857],
    [0.5, 0.98090376],
    [0.62031509, 0.96054857],
    [0.72673443, 0.89158001],
    [0.82335836, 0.80549044],
    [0.90243736, 0.70239563],
    [0.95414744, 0.5833164],
    [0.9819403, 0.45481963],
    [0.99657961, 0.32351251],
    [1., 0.19162136],
    [0.09431708, 0.09303366],
    [0.156603, 0.03609577],
    [0.24499027, 0.01909624],
    [0.33640482, 0.03208252],
    [0.4217254, 0.06810575],
    [0.5782746, 0.06810575],
    [0.66359518, 0.03208252],
    [0.75500973, 0.01909624],
    [0.843397, 0.03609577],
    [0.90568292, 0.09303366],
    [0.5, 0.17077372],
    [0.5, 0.25582131],
    [0.5, 0.34018057],
    [0.5, 0.42711253],
    [0.39750296, 0.48606286],
    [0.44679426, 0.50396655],
    [0.5, 0.51947823],
    [0.55320574, 0.50396655],
    [0.60249704, 0.48606286],
    [0.19547876, 0.18200029],
    [0.2495543, 0.15079845],
    [0.31450698, 0.151619],
    [0.37155765, 0.19479588],
    [0.31041216, 0.20588273],
    [0.2457658, 0.20499598],
    [0.62844235, 0.19479588],
    [0.68549302, 0.151619],
    [0.7504457, 0.15079845],
    [0.80452124, 0.18200029],
    [0.7542342, 0.20499598],
    [0.68958784, 0.20588273],
    [0.30589462, 0.64454894],
    [0.37753891, 0.6160173],
    [0.44988263, 0.60094314],
    [0.5, 0.61354581],
    [0.55011737, 0.60094314],
    [0.62246109, 0.6160173],
    [0.69410538, 0.64454894],
    [0.62477271, 0.71392546],
    [0.55454877, 0.74405802],
    [0.5, 0.74971382],
    [0.44545123, 0.74405802],
    [0.37522729, 0.71392546],
    [0.33558146, 0.6482952],
    [0.44915626, 0.64312397],
    [0.5, 0.64850295],
    [0.55084374, 0.64312397],
    [0.66441854, 0.6482952],
    [0.55181764, 0.67991149],
    [0.5, 0.68607805],
    [0.44818236, 0.67991149]], dtype=np.float32)

FLIP_300W = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29, 30, 35, 34, 33,
             32, 31, 45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63, 62, 61, 60, 67, 66, 65]


class Container300W(BaseContainer):
    def __init__(self,
                 path,
                 cache_path,
                 mmap_mode='r',
                 name=None,
                 cache_size=224,
                 cache_padding=0.5):
        super().__init__(path, cache_path, mmap_mode, name)
        self.cache_size = cache_size
        self.cache_padding = cache_padding

    @property
    def config(self):
        return {
            'path': str(self.path),
            'cache_size': self.cache_size,
            'cache_padding': self.cache_padding,
        }

    def make_cache(self):
        mean_shape_padding = ((MEANSHAPE_300W - 0.5) *
                              (1 - self.cache_padding) + 0.5) * self.cache_size

        def func(image_path):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR |
                               cv2.IMREAD_IGNORE_ORIENTATION)
            lmk = np.genfromtxt(Path(image_path).with_suffix('.pts'),
                                skip_header=3, skip_footer=1).astype(np.float32) - 1.0

            transform = umeyama(lmk, mean_shape_padding).astype(np.float32)
            image = cv2.warpAffine(image, fix_opencv_(transform)[
                :2, :], (self.cache_size, self.cache_size), flags=cv2.INTER_AREA)
            lmk = lmk @ transform[:2, :2].T + transform[:2, 2]
            return (image, lmk, )

        self.cache_path.mkdir(exist_ok=True, parents=True)
        return multi_process(self.parsing_dataset(),
                             func,
                             self.cache_path / '{}.npy'.format(self.name),
                             ['image', 'landmark'])

    def parsing_dataset(self):
        images_path = []
        for p in self.path:
            images_path += list(Path(p).rglob('*.jpg'))
            images_path += list(Path(p).rglob('*.png'))
        return list(map(str, images_path))


class Dataset300W(Dataset):
    def __init__(self, face_container, output_size=112, padding=0.3, augment=False):
        super().__init__()
        self.face_container = face_container
        self.output_size = output_size
        self.padding = padding
        self.augment = augment
        self.mean_shape = ((MEANSHAPE_300W - 0.5) *
                           (1 - padding) + 0.5) * output_size

    def __len__(self):
        return len(self.face_container)

    def __getitem__(self, index):
        return self.transform(self.face_container[index])

    def transform(self, sample):
        image, landmark = sample['image'], sample['landmark']

        trans = umeyama(landmark, self.mean_shape)
        if self.augment:
            # flip
            if random.choice((True, False)):
                trans = from_hflip(self.output_size) @ trans
                landmark = landmark[FLIP_300W]
            angle = np.random.normal(0, 20.0)
            scale = np.random.normal(1.0, 0.25)
            translate = np.random.normal(0.0, 0.2, 2) * self.output_size

            trans = from_translate(translate) @ from_center_rotate(
                (self.output_size / 2, self.output_size / 2), angle, scale) @ trans

        landmark = landmark @ trans[:2, :2].T + trans[:2, 2]
        image = cv2.warpAffine(image, fix_opencv_(
            trans)[:2, :], (self.output_size, self.output_size), flags=cv2.INTER_AREA)
        return {'image': image, 'landmark': landmark}
