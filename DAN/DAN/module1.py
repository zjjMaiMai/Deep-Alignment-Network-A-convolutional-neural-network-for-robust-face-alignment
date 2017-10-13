import numpy as np
import tensorflow as tf
import cv2

pixels = [(x, y) for x in range(112) for y in range(112)]
pixels = np.array(pixels, dtype=np.float32)
pixelsT = tf.constant(pixels)

Transform = tf.constant([0.7,0.7,-0.7,0.7,
                         50,10],tf.float32)
A = tf.reshape(Transform[0:4],[2,2])
t = Transform[4:]

outPixels = tf.tensordot(pixelsT - t,tf.matrix_inverse(A),1)
outPixels = tf.clip_by_value(outPixels,0,111)

Images = tf.placeholder(tf.uint8,[112,112])

Image = tf.image.convert_image_dtype(Images,tf.float32)
Idx = tf.cast(tf.round(outPixels),tf.int64)
outPixelsMaxMin = Idx + [1, 0]
outPixelsMinMax = Idx + [0, 1]
outPixelsMaxMax = Idx + [1, 1]

d = outPixels - tf.cast(Idx,tf.float32)
dx = d[:,0]
dy = d[:,1]

Ret = tf.gather_nd(tf.reshape((1 - dx) * (1 - dy),[112,112]) * Image,Idx) + tf.gather_nd(tf.reshape(dx * (1 - dy),[112,112]) * Image,Idx) + tf.gather_nd(tf.reshape((1 - dx) * dy,[112,112]) * Image,Idx) + tf.gather_nd(tf.reshape(dx * dy,[112,112]) * Image,Idx)



ti = cv2.imread('1.jpg',cv2.IMREAD_GRAYSCALE)
ti = cv2.resize(ti,(112,112))
ti = np.reshape(ti,(112,112))

print(cv2.getRotationMatrix2D((0,0),45,1))


with tf.Session() as Sess:
    Sess.run(tf.global_variables_initializer())
    Img = Sess.run(Image,{Images:ti}).reshape((112,112,1))
    cv2.imshow('test',Img)
    Ret = Sess.run(Ret,{Images:ti}).reshape((112,112,1))
    cv2.imshow('Ret',Ret)
    cv2.waitKey(-1)
