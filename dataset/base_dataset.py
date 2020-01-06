from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import json
import tqdm
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed


def get_tuple_dtype(obj: tuple, name_list=None):
    """get input object type in numpy.

    :param obj: input tuple of object
    :param name_list: name of each feature, if None, use f0, f1, ...

    :return: struct numpy array's dtype of tuple object
    """
    if name_list is None:
        name_list = ['f{}'.format(i) for i in range(len(obj))]
    assert len(name_list) == len(
        obj), "name_list {} obj {}".format(len(name_list), len(obj))

    dtype = []
    for value, name in zip(obj, name_list):
        assert isinstance(value, np.ndarray)
        dtype.append((name, value.dtype, value.shape))
    dtype = np.dtype(dtype)
    return dtype


def multi_process(list_of_sample, func, output_path, name_list=None, dtype=None, shape=None):
    """multiprocess on list of sample

    :param list_of_sample: list of data
    :param func: process function
    :param output_path: cache output path
    :param name_list: see `_get_dtype` function
    :param dtype & shape: dtype and shape by func return

    :return: memory map numpy struct array
    """
    list_len = 1
    if dtype is None or shape is None:
        first_sample = func(list_of_sample[0])
        if isinstance(first_sample, np.ndarray):
            shape = (len(list_of_sample), *first_sample.shape)
            dtype = first_sample.dtype
        elif isinstance(first_sample, tuple):
            shape = (len(list_of_sample), )
            dtype = get_tuple_dtype(first_sample, name_list)
        elif isinstance(first_sample, list):
            list_len = len(first_sample)
            shape = (len(list_of_sample) * list_len, )
            dtype = get_tuple_dtype(first_sample[0], name_list)
        else:
            raise NotImplementedError('Unsupport Type!')

    out_mmap = np.lib.format.open_memmap(output_path, mode='w+',
                                         dtype=dtype, shape=shape)

    def _func(idx, x):
        out_mmap[idx * list_len:(idx + 1) * list_len] = func(x)

    Parallel(n_jobs=-1)(delayed(_func)(idx, sample)
                        for idx, sample in enumerate(tqdm.tqdm(list_of_sample)))
    return out_mmap


class BaseContainer(object):
    """ base class of all container.
    """

    def __init__(self,
                 path,
                 cache_path,
                 mmap_mode='r',
                 name=None):
        self.path = [Path(p) for p in path]
        self.cache_path = Path(cache_path)
        self.mmap_mode = mmap_mode
        self.name = self.__class__.__name__ if name is None else name
        self._data = None

    @property
    def config(self):
        raise NotImplementedError(
            "{}: Method not implemented".format(self.__class__.__name__))

    def make_cache(self):
        raise NotImplementedError(
            "{}: Method not implemented".format(self.__class__.__name__))

    @property
    def data(self):
        if self._data is not None:
            return self._data

        self._data = self.load_or_make_cache()
        return self._data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def load_or_make_cache(self):
        try:
            with open(self.cache_path / (self.name + '_cache_config.json'), 'r') as fp:
                config = json.load(fp)
                assert config == self.config
            assert (self.cache_path /
                    '{}.npy'.format(self.name)).exists()
        except:
            self.make_cache()
            with open(self.cache_path / (self.name + '_cache_config.json'), 'w+') as fp:
                json.dump(self.config, fp)

        return np.load(self.cache_path / '{}.npy'.format(self.name),
                       mmap_mode=self.mmap_mode)
