import unittest
import tensorflow as tf
import numpy as np

from dan.dan_model import _umeyama_tf, _transfrom_lmk, _transfrom_img
from utils.transform.umeyama import umeyama
from utils.transform.trans2d import from_center_rotate


class TestDANFunc(unittest.TestCase):
    def test_umeyama_tf(self):
        src = np.random.rand(128, 16, 2)
        dst = np.random.rand(128, 16, 2)
        out_tf_op = _umeyama_tf(src, dst)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            out_tf = sess.run(out_tf_op)
        out_np = np.array([umeyama(s, d) for s, d in zip(src, dst)])

        self.assertTrue(np.allclose(out_tf, out_np))

    def test_transfrom_lmk(self):
        def transfrom_lmk_np(lmk, transfrom, inv=False):
            if inv:
                transfrom = np.linalg.inv(transfrom)
            ret = np.array([l @ t[:2, :2].T + t[:2, 2]
                            for l, t in zip(lmk, transfrom)])
            return ret

        src = np.random.rand(128, 16, 2)
        transfrom = np.array(
            [from_center_rotate(
                tuple(np.random.rand(2)),
                np.random.uniform(-180, 180),
                np.random.uniform(0.1, 1.0)) for _ in range(src.shape[0])])

        out_tf_op = _transfrom_lmk(src, transfrom, inv=False)
        out_tf_inv_op = _transfrom_lmk(src, transfrom, inv=True)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            out_tf, out_tf_inv = sess.run([out_tf_op, out_tf_inv_op])

        out_np = transfrom_lmk_np(src, transfrom, inv=False)
        out_np_inv = transfrom_lmk_np(src, transfrom, inv=True)

        self.assertTrue(np.allclose(out_tf, out_np))
        self.assertTrue(np.allclose(out_tf_inv, out_np_inv))

    def test_transfrom_img(self):
        import cv2

        def transfrom_img_cv2(img, transfrom, inv=False):
            img_size = (img.shape[2], img.shape[1])
            if inv:
                transfrom = np.linalg.inv(transfrom)
            ret = np.array([cv2.warpAffine(
                i, t[:2, :], img_size, flags=cv2.INTER_NEAREST) for i, t in zip(img, transfrom)])
            return ret

        img = np.zeros((32, 64, 64, 3), dtype=np.float32)
        img[:, 16:48, 16:48, :] = 1
        transfrom = np.array(
            [from_center_rotate(
                tuple(np.random.uniform(0, 64, size=2)),
                np.random.uniform(-180, 180),
                np.random.uniform(0.1, 1.0)).astype(np.float32) for _ in range(img.shape[0])])

        out_tf_op = _transfrom_img(img, transfrom, inv=False)
        out_tf_inv_op = _transfrom_img(img, transfrom, inv=True)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            out_tf, out_tf_inv = sess.run([out_tf_op, out_tf_inv_op])

        out_cv2 = transfrom_img_cv2(img, transfrom, inv=False)
        out_cv2_inv = transfrom_img_cv2(img, transfrom, inv=True)

        self.assertTrue(np.sum(np.abs(out_cv2 - out_tf)) < 5 * img.shape[0])
        self.assertTrue(np.sum(np.abs(out_cv2_inv - out_tf_inv))
                        < 5 * img.shape[0])


if __name__ == "__main__":
    unittest.main()
