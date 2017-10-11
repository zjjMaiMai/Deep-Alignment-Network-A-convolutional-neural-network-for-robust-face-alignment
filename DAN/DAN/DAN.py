import tensorflow as tf
import numpy as np  
import matplotlib.pyplot as plt  
import cv2
import pickle
import matplotlib.pyplot as plt

from scipy import ndimage
import numpy as np
import glob
from os import path
import math

class ImageServer(object):
    def __init__(self, imgSize=[112, 112], frameFraction=0.25, initialization='box', color=False):
        self.origLandmarks = []
        self.filenames = []
        self.mirrors = []
        self.meanShape = np.array([])

        self.meanImg = np.array([])
        self.stdDevImg = np.array([])

        self.perturbations = []

        self.imgSize = imgSize
        self.frameFraction = frameFraction
        self.initialization = initialization
        self.color = color

        self.boundingBoxes = []

    @staticmethod
    def Load(filename):
        imageServer = ImageServer()
        arrays = np.load(filename)
        imageServer.__dict__.update(arrays)

        if (len(imageServer.imgs.shape) == 3):
            imageServer.imgs = imageServer.imgs[:, np.newaxis]

        return imageServer

    def Save(self, datasetDir, filename=None):
        if filename is None:
            filename = "dataset_nimgs={0}_perturbations={1}_size={2}".format(len(self.imgs), list(self.perturbations), self.imgSize)
            if self.color:
                filename += "_color={0}".format(self.color)
            filename += ".npz"

        arrays = {key:value for key, value in self.__dict__.items() if not key.startswith('__') and not callable(key)}
        np.savez(datasetDir + filename, **arrays)

    def PrepareData(self, imageDirs, boundingBoxFiles, meanShape, startIdx, nImgs, mirrorFlag):
        filenames = []
        landmarks = []
        boundingBoxes = []


        for i in range(len(imageDirs)):
            filenamesInDir = glob.glob(imageDirs[i] + "*.jpg")
            filenamesInDir += glob.glob(imageDirs[i] + "*.png")

            if boundingBoxFiles is not None:
                boundingBoxDict = pickle.load(open(boundingBoxFiles[i], 'rb'))

            for j in range(len(filenamesInDir)):
                filenames.append(filenamesInDir[j])
                
                ptsFilename = filenamesInDir[j][:-3] + "pts"
                landmarks.append(utils.loadFromPts(ptsFilename))

                if boundingBoxFiles is not None:
                    basename = path.basename(filenamesInDir[j])
                    boundingBoxes.append(boundingBoxDict[basename])
                

        filenames = filenames[startIdx : startIdx + nImgs]
        landmarks = landmarks[startIdx : startIdx + nImgs]
        boundingBoxes = boundingBoxes[startIdx : startIdx + nImgs]

        mirrorList = [False for i in range(nImgs)]
        if mirrorFlag:     
            mirrorList = mirrorList + [True for i in range(nImgs)]
            filenames = np.concatenate((filenames, filenames))

            landmarks = np.vstack((landmarks, landmarks))
            boundingBoxes = np.vstack((boundingBoxes, boundingBoxes))       

        self.origLandmarks = landmarks
        self.filenames = filenames
        self.mirrors = mirrorList
        self.meanShape = meanShape
        self.boundingBoxes = boundingBoxes

    def LoadImages(self):
        self.imgs = []
        self.initLandmarks = []
        self.gtLandmarks = []

        for i in range(len(self.filenames)):
            img = ndimage.imread(self.filenames[i])

            if self.color:
                if len(img.shape) == 2:
                    img = np.dstack((img, img, img))
            else:
                if len(img.shape) > 2:
                    img = np.mean(img, axis=2)
            img = img.astype(np.uint8)

            if self.mirrors[i]:
                self.origLandmarks[i] = utils.mirrorShape(self.origLandmarks[i], img.shape)
                img = np.fliplr(img)

            if self.color:
                img = np.transpose(img, (2, 0, 1))
            else:
                img = img[np.newaxis]

            groundTruth = self.origLandmarks[i]

            if self.initialization == 'rect':
                bestFit = utils.bestFitRect(groundTruth, self.meanShape)
            elif self.initialization == 'similarity':
                bestFit = utils.bestFit(groundTruth, self.meanShape)
            elif self.initialization == 'box':
                bestFit = utils.bestFitRect(groundTruth, self.meanShape, box=self.boundingBoxes[i])

            self.imgs.append(img)
            self.initLandmarks.append(bestFit)
            self.gtLandmarks.append(groundTruth)

        self.initLandmarks = np.array(self.initLandmarks)
        self.gtLandmarks = np.array(self.gtLandmarks)    

    def GeneratePerturbations(self, nPerturbations, perturbations):
        self.perturbations = perturbations
        meanShapeSize = max(self.meanShape.max(axis=0) - self.meanShape.min(axis=0))
        destShapeSize = min(self.imgSize) * (1 - 2 * self.frameFraction)
        scaledMeanShape = self.meanShape * destShapeSize / meanShapeSize

        newImgs = []  
        newGtLandmarks = []
        newInitLandmarks = []           

        translationMultX, translationMultY, rotationStdDev, scaleStdDev = perturbations

        rotationStdDevRad = rotationStdDev * np.pi / 180         
        translationStdDevX = translationMultX * (scaledMeanShape[:, 0].max() - scaledMeanShape[:, 0].min())
        translationStdDevY = translationMultY * (scaledMeanShape[:, 1].max() - scaledMeanShape[:, 1].min())
        print("Creating perturbations of " + str(self.gtLandmarks.shape[0]) + " shapes")

        for i in range(self.initLandmarks.shape[0]):
            print(i)
            for j in range(nPerturbations):
                tempInit = self.initLandmarks[i].copy()

                angle = np.random.normal(0, rotationStdDevRad)
                offset = [np.random.normal(0, translationStdDevX), np.random.normal(0, translationStdDevY)]
                scaling = np.random.normal(1, scaleStdDev)

                R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])     
            
                tempInit = tempInit + offset
                tempInit = (tempInit - tempInit.mean(axis=0)) * scaling + tempInit.mean(axis=0)            
                tempInit = np.dot(R, (tempInit - tempInit.mean(axis=0)).T).T + tempInit.mean(axis=0)

                tempImg, tempInit, tempGroundTruth = self.CropResizeRotate(self.imgs[i], tempInit, self.gtLandmarks[i])                

                newImgs.append(tempImg)
                newInitLandmarks.append(tempInit)
                newGtLandmarks.append(tempGroundTruth)

        self.imgs = np.array(newImgs)
        self.initLandmarks = np.array(newInitLandmarks)
        self.gtLandmarks = np.array(newGtLandmarks)

    def CropResizeRotateAll(self):
        newImgs = []  
        newGtLandmarks = []
        newInitLandmarks = []   
        
        for i in range(self.initLandmarks.shape[0]):
            tempImg, tempInit, tempGroundTruth = self.CropResizeRotate(self.imgs[i], self.initLandmarks[i], self.gtLandmarks[i])

            newImgs.append(tempImg)
            newInitLandmarks.append(tempInit)
            newGtLandmarks.append(tempGroundTruth)

        self.imgs = np.array(newImgs)
        self.initLandmarks = np.array(newInitLandmarks)
        self.gtLandmarks = np.array(newGtLandmarks)  

    def NormalizeImages(self, imageServer=None):
        self.imgs = self.imgs.astype(np.float32)

        if imageServer is None:
            self.meanImg = np.mean(self.imgs, axis=0)
        else:
            self.meanImg = imageServer.meanImg

        self.imgs = self.imgs - self.meanImg
        
        if imageServer is None:
            self.stdDevImg = np.std(self.imgs, axis=0)
        else:
            self.stdDevImg = imageServer.stdDevImg
        
        self.imgs = self.imgs / self.stdDevImg

        from matplotlib import pyplot as plt  

        meanImg = self.meanImg - self.meanImg.min()
        meanImg = 255 * meanImg / meanImg.max()  
        meanImg = meanImg.astype(np.uint8)   
        if self.color:
            plt.imshow(np.transpose(meanImg, (1, 2, 0)))
        else:
            plt.imshow(meanImg[0], cmap=plt.cm.gray)
        plt.savefig("../meanImg.jpg")
        plt.clf()

        stdDevImg = self.stdDevImg - self.stdDevImg.min()
        stdDevImg = 255 * stdDevImg / stdDevImg.max()  
        stdDevImg = stdDevImg.astype(np.uint8)   
        if self.color:
            plt.imshow(np.transpose(stdDevImg, (1, 2, 0)))
        else:
            plt.imshow(stdDevImg[0], cmap=plt.cm.gray)
        plt.savefig("../stdDevImg.jpg")
        plt.clf()

    def CropResizeRotate(self, img, initShape, groundTruth):
        meanShapeSize = max(self.meanShape.max(axis=0) - self.meanShape.min(axis=0))
        destShapeSize = min(self.imgSize) * (1 - 2 * self.frameFraction)

        scaledMeanShape = self.meanShape * destShapeSize / meanShapeSize

        destShape = scaledMeanShape.copy() - scaledMeanShape.mean(axis=0)
        offset = np.array(self.imgSize[::-1]) / 2
        destShape += offset

        A, t = utils.bestFit(destShape, initShape, True)
    
        A2 = np.linalg.inv(A)
        t2 = np.dot(-t, A2)

        outImg = np.zeros((img.shape[0], self.imgSize[0], self.imgSize[1]), dtype=img.dtype)
        for i in range(img.shape[0]):
            outImg[i] = ndimage.interpolation.affine_transform(img[i], A2, t2[[1, 0]], output_shape=self.imgSize)

        initShape = np.dot(initShape, A) + t

        groundTruth = np.dot(groundTruth, A) + t
        return outImg, initShape, groundTruth


def PredictErr(GroudTruth,Predict):
    Gt = tf.reshape(GroudTruth,[-1,68,2])
    Ot = tf.reshape(Predict,[-1,68,2])

    def MeanErr(flt,Mix):
        Current,Gt = Mix
        MeanErr = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.subtract(Current, Gt) ** 2,1)))
        EyeDistance = tf.norm(tf.reduce_mean(Gt[36:42],0) - tf.reduce_mean(Gt[42:48],0))
        return MeanErr / EyeDistance

    return tf.scan(fn=MeanErr,elems=[Ot,Gt],initializer=0.0)

Feed_dict = {}
Ret_dict = {}

#Procrustes analysis
def transformation_from_points(points1, points2):
    points1 = np.asmatrix(points1.copy().reshape((-1,2)))
    points2 = np.asmatrix(points2.copy().reshape((-1,2)))

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return np.asarray(np.vstack([np.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T))]))

def Stage1(InputS0=None):
    InputImage = tf.placeholder(tf.float32,[None,112,112,1])
    GroundTruth = tf.placeholder(tf.float32,[None,136])
    S1_isTrain = tf.placeholder(tf.bool)

    S0 = tf.constant(InputS0)

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
    S1_Fc2 = tf.layers.dense(S1_Fc1,136)

    S1_Ret = S1_Fc2 + S0

    S1_Cost = tf.reduce_mean(PredictErr(GroundTruth,S1_Ret))
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        S1_Optimizer = tf.train.AdamOptimizer(0.001).minimize(S1_Cost)

    Feed_dict['S1_InputImage'] = InputImage
    Feed_dict['S1_GroundTruth'] = GroundTruth
    Feed_dict['S1_isTrain'] = S1_isTrain

    Ret_dict['S1_Ret'] = S1_Ret
    Ret_dict['S1_Cost'] = S1_Cost
    Ret_dict['S1_Feature'] = S1_Fc1
    Ret_dict['S1_Train'] = S1_Optimizer

def Stage2():
    S2_InputImage = tf.placeholder(tf.float32,[None,112,112,1])
    S2_GroundTruth = tf.placeholder(tf.float32,[None,136])
    S2_isTrain = tf.placeholder(tf.bool)

    S2_InputFeature = tf.placeholder(tf.float32,[None,256])
    S2_InputHeatmap = tf.placeholder(tf.float32,[None,112,112,1])
    S2_InputInitLandmark = tf.placeholder(tf.float32,[None,136])

    S2_Feature = tf.reshape(tf.layers.dense(S2_InputFeature,56 * 56,activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),(-1,56,56,1))
    S2_FeatureUpScale = tf.image.resize_images(S2_Feature,(112,112))

    S2_ConcatInput = tf.layers.batch_normalization(tf.concat([S2_InputImage,S2_InputHeatmap,S2_FeatureUpScale],2),training=S2_isTrain)
    S2_Conv1a = tf.layers.batch_normalization(tf.layers.conv2d(S2_ConcatInput,64,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
    S2_Conv1b = tf.layers.batch_normalization(tf.layers.conv2d(S2_Conv1a,64,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
    S2_Pool1 = tf.layers.max_pooling2d(S2_Conv1b,2,2,padding='same')

    S2_Conv2a = tf.layers.batch_normalization(tf.layers.conv2d(S2_Pool1,128,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
    S2_Conv2b = tf.layers.batch_normalization(tf.layers.conv2d(S2_Conv2a,128,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
    S2_Pool2 = tf.layers.max_pooling2d(S2_Conv2b,2,2,padding='same')

    S2_Conv3a = tf.layers.batch_normalization(tf.layers.conv2d(S2_Pool2,256,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
    S2_Conv3b = tf.layers.batch_normalization(tf.layers.conv2d(S2_Conv3a,256,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
    S2_Pool3 = tf.layers.max_pooling2d(S2_Conv3b,2,2,padding='same')

    S2_Conv4a = tf.layers.batch_normalization(tf.layers.conv2d(S2_Pool3,512,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
    S2_Conv4b = tf.layers.batch_normalization(tf.layers.conv2d(S2_Conv4a,512,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
    S2_Pool4 = tf.layers.max_pooling2d(S2_Conv4b,2,2,padding='same')

    S2_Pool4_Flat = tf.contrib.layers.flatten(S2_Pool4)
    S2_DropOut = tf.layers.dropout(S2_Pool4_Flat,0.5,training=S2_isTrain)

    S2_Fc1 = tf.layers.batch_normalization(tf.layers.dense(S2_DropOut,256,activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S2_isTrain)
    S2_Fc2 = tf.layers.dense(S2_Fc1,136)

    S2_Ret = S2_Fc2 + S2_InputInitLandmark

    S2_Cost = tf.reduce_mean(PredictErr(S2_GroundTruth,S2_Ret))

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        S2_Optimizer = tf.train.AdamOptimizer(0.001).minimize(S2_Cost)

    Feed_dict['S2_InputImage'] = S2_InputImage
    Feed_dict['S2_GroundTruth'] = S2_GroundTruth
    Feed_dict['S2_isTrain'] = S2_isTrain

    Ret_dict['S2_Ret'] = S2_Ret
    Ret_dict['S2_Cost'] = S2_Cost
    Ret_dict['S2_Feature'] = S2_Fc1
    Ret_dict['S2_Train'] = S2_Optimizer


def ConnectionLayer(Image,InputLandmark,MeanShape):
    M = transformation_from_points(InputLandmark,MeanShape)

    Is = np.reshape(InputLandmark,(68,1,2))
    Ms = np.reshape(MeanShape,(68,1,2))

    OutImage = cv2.warpAffine(Image,M,(112,112))
    OutShape = cv2.transform(Is,M)
    OutShape = np.reshape(OutShape,-1)
       
    HeatMap = np.full((112 + 16,112 + 16),100000,dtype=np.float32)
    for i in range(68):
        LandX = OutShape[i * 2]
        LandY = OutShape[i * 2 + 1]

        for y in range(-8,8):
            for x in range(-8,8):
                dis = (x ** 2 + y ** 2) ** 0.5
                Y = int(round(LandY + y))
                X = int(round(LandX + x))
                if HeatMap[Y][X] >= dis: 
                    HeatMap[Y][X] = dis

    HeatMap = 1 / (1 + HeatMap[0:112,0:112])
    return OutImage,OutShape,M,HeatMap
    

trainSet = ImageServer.Load("C:\\Users\\ZhaoHang\\Desktop\\DeepAlignmentNetwork-master\\data\\dataset_nimgs=60960_perturbations=[0.2, 0.2, 20, 0.25]_size=[112, 112].npz")
validationSet = ImageServer.Load("C:\\Users\\ZhaoHang\\Desktop\\DeepAlignmentNetwork-master\\data\\dataset_nimgs=100_perturbations=[]_size=[112, 112].npz")


def getLabelsForDataset(imageServer):
    nSamples = imageServer.gtLandmarks.shape[0]
    nLandmarks = imageServer.gtLandmarks.shape[1]

    y = np.zeros((nSamples, nLandmarks, 2), dtype=np.float32)
    #y[:, 0] = imageServer.initLandmarks
    y = imageServer.gtLandmarks

    return y

nSamples = trainSet.gtLandmarks.shape[0]
imageHeight = trainSet.imgSize[0]
imageWidth = trainSet.imgSize[1]
nChannels = trainSet.imgs.shape[1]

Xtrain = trainSet.imgs
Xvalid = validationSet.imgs

Ytrain = getLabelsForDataset(trainSet)
Yvalid = getLabelsForDataset(validationSet)

testIdxsTrainSet = range(len(Xvalid))
testIdxsValidSet = range(len(Xvalid))

meanImg = trainSet.meanImg
stdDevImg = trainSet.stdDevImg
initLandmarks = trainSet.initLandmarks[0]

#File = open("ImageData.pkl","rb")
#I = pickle.load(File)
#G = pickle.load(File)
#Ti = pickle.load(File)
#Tg = pickle.load(File)
I = Xtrain
G = Ytrain
Ti = Xvalid
Tg = Yvalid
 

I = np.reshape(I,(-1,112,112,1))
Ti = np.reshape(Ti,(-1,112,112,1))

G = np.reshape(G,[-1,136]).astype(np.float32)
Tg = np.reshape(Tg,[-1,136]).astype(np.float32)
MeanShape = np.reshape(initLandmarks,[-1,136]).astype(np.float32)

Stage1(MeanShape)
Stage2()

with tf.Session() as Sess:
    Sess.run(tf.global_variables_initializer())
    for w in range(100):
        Count = 0
        RandomIdx = np.random.choice(I.shape[0],64,False)
        while Count < I.shape[0] // 64:
            _,S1Ret = Sess.run([Ret_dict['S1_Train'],Ret_dict['S1_Ret']],{Feed_dict['S1_InputImage']:I[RandomIdx],
                                                                          Feed_dict['S1_GroundTruth']:G[RandomIdx],
                                                                          Feed_dict['S1_isTrain']:True})

            if Count % 100 == 0:
                TestErr = Sess.run(Ret_dict['S1_Cost'],{Feed_dict['S1_InputImage']:Ti,
                                                        Feed_dict['S1_GroundTruth']:Tg,
                                                        Feed_dict['S1_isTrain']:False})
                BatchErr = Sess.run(Ret_dict['S1_Cost'],{Feed_dict['S1_InputImage']:I[RandomIdx],
                                                         Feed_dict['S1_GroundTruth']:G[RandomIdx],
                                                         Feed_dict['S1_isTrain']:False})
                print(w,Count,'TestErr:',TestErr,' BatchErr:',BatchErr)
            Count += 1
    saver = tf.train.Saver()
    saver.save(Sess,'./Model/Model')
