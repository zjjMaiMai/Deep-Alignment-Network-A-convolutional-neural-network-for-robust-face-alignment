
import os
import sys
import glob

import tensorflow as tf
import cv2
import numpy as np


imgs_mean = np.loadtxt("D:\\Dataset\\Test1\\imgs_mean.ptv",dtype=np.float32,delimiter=',').reshape([112,112,1])
imgs_std = np.loadtxt("D:\\Dataset\\Test1\\imgs_std.ptv",dtype=np.float32,delimiter=',').reshape([112,112,1])
meanshape = np.loadtxt("D:\\Dataset\\Test1\\mean_shape.ptv",dtype=np.float32,delimiter=',').reshape([68,2])

print(imgs_mean,imgs_std,meanshape)

def model_fn(features, labels, mode, params):
    features = (tf.reshape(features,[-1,112,112,1]) - imgs_mean) / imgs_std
    labels = tf.reshape(labels,[-1,68,2])



    LANDMARK = 68

    def NormRmse(GroudTruth, Prediction):
        Gt = tf.reshape(GroudTruth, [-1, LANDMARK, 2])
        Pt = tf.reshape(Prediction, [-1, LANDMARK, 2])
        loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(Gt, Pt), 2)), 1)
        norm = tf.norm(tf.reduce_mean(Gt[:, 36:42, :],1) - tf.reduce_mean(Gt[:, 42:48, :],1), axis=1)

        return loss/norm

    with tf.variable_scope('Stage1'):
        InputImage = features
        S1_isTrain = (mode == tf.estimator.ModeKeys.TRAIN)

        S1_Conv1a = tf.layers.batch_normalization(tf.layers.conv2d(InputImage,64,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        S1_Conv1b = tf.layers.batch_normalization(tf.layers.conv2d(S1_Conv1a,64,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        S1_Pool1 = tf.layers.max_pooling2d(S1_Conv1b,2,2,padding='same')

        S1_Conv2a = tf.layers.batch_normalization(tf.layers.conv2d(S1_Pool1,128,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        S1_Conv2b = tf.layers.batch_normalization(tf.layers.conv2d(S1_Conv2a,128,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        S1_Pool2 = tf.layers.max_pooling2d(S1_Conv2b,2,2,padding='same')

        S1_Conv3a = tf.layers.batch_normalization(tf.layers.conv2d(S1_Pool2,256,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        S1_Conv3b = tf.layers.batch_normalization(tf.layers.conv2d(S1_Conv3a,256,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        S1_Pool3 = tf.layers.max_pooling2d(S1_Conv3b,2,2,padding='same')

        S1_Conv4a = tf.layers.batch_normalization(tf.layers.conv2d(S1_Pool3,512,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        S1_Conv4b = tf.layers.batch_normalization(tf.layers.conv2d(S1_Conv4a,512,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        S1_Pool4 = tf.layers.max_pooling2d(S1_Conv4b,2,2,padding='same')

        S1_Pool4_Flat = tf.contrib.layers.flatten(S1_Pool4)
        S1_DropOut = tf.layers.dropout(S1_Pool4_Flat,0.5,training=S1_isTrain)

        S1_Fc1 = tf.layers.batch_normalization(tf.layers.dense(S1_DropOut,256,activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain)
        S1_Fc2 = tf.layers.dense(S1_Fc1,LANDMARK * 2)

        S1_Ret = tf.reshape(S1_Fc2,[-1,LANDMARK,2]) + meanshape

        S1_Cost = tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.squared_difference(labels,S1_Ret),-1)),-1) / tf.sqrt(tf.reduce_sum(tf.squared_difference(tf.reduce_max(labels,1),tf.reduce_min(labels,1)),-1)))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'Stage1')):
            S1_Optimizer = tf.train.AdamOptimizer(0.001).minimize(S1_Cost,global_step=tf.train.get_or_create_global_step(),var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"Stage1"))

    return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions=S1_Ret,
    loss=S1_Cost,
    train_op=S1_Optimizer)


def get_filenames(data_dir):
    listext = ['*.png']

    imagelist = []
    for ext in listext:
        p = os.path.join(data_dir, ext)
        imagelist.extend(glob.glob(p))

    ptslist = []
    for image in imagelist:
        ptslist.append(os.path.splitext(image)[0] + ".ptv")

    return imagelist, ptslist

def input_fn():
    trainimg,traingt = get_filenames("D:\\Dataset\\Test1\\")

    def decode_img_pts(img,pts):
        #img = np.load(img.decode())
        img = cv2.imread(img.decode(),cv2.IMREAD_GRAYSCALE)
        pts = np.loadtxt(pts.decode(),dtype=np.float32,delimiter=',')
        return img.astype(np.float32),pts.astype(np.float32)

    img = tf.data.Dataset.from_tensor_slices(trainimg)
    pts = tf.data.Dataset.from_tensor_slices(traingt)

    dataset = tf.data.Dataset.zip((img, pts))
    dataset = dataset.prefetch(64)
    dataset = dataset.shuffle(1000)
    dataset = dataset.repeat()
    dataset = dataset.map(lambda img,pts:tuple(tf.py_func(decode_img_pts,[img,pts],[tf.float32,tf.float32])))
    dataset = dataset.batch(64)
    dataset = dataset.prefetch(1)

    return dataset


estimator = tf.estimator.Estimator(model_fn=model_fn,model_dir='./ModelDir',params = {'la':64})
estimator.train(input_fn=input_fn)