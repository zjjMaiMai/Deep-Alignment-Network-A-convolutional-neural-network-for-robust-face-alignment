import tensorflow as tf
import numpy as np  
import itertools
import cv2
import time
from scipy import ndimage

IMGSIZE = 112
LANDMARK = 68

LEFTEYE_START = 0
LEFTEYE_END = 0
RIGHTEYE_START = 0
RIGHTEYE_END = 0

if LANDMARK == 36:
    LEFTEYE_START = 8
    LEFTEYE_END = 12
    RIGHTEYE_START = 12
    RIGHTEYE_END = 16

if LANDMARK == 68:
    LEFTEYE_START = 36
    LEFTEYE_END = 42
    RIGHTEYE_START = 42
    RIGHTEYE_END = 48


#test Good
def GetAffineParam(ShapesFrom,ShapeTo):
    def Do(From,To):
        destination = tf.reshape(To,[-1,2])
        source = tf.reshape(From,[-1,2])

        destMean = tf.reduce_mean(destination,0)
        srcMean = tf.reduce_mean(source,0)

        srcCenter = source - srcMean
        destCenter = destination - destMean

        srcVec = tf.reshape(srcCenter,[-1])
        destVec = tf.reshape(destCenter,[-1])

        Temp = tf.norm(srcVec) ** 2
        a = tf.tensordot(srcVec,destVec,1) / Temp
        b = 0

        SrcX = tf.reshape(srcVec,[-1,2])[:,0]
        SrcY = tf.reshape(srcVec,[-1,2])[:,1]
        DestX = tf.reshape(destVec,[-1,2])[:,0]
        DestY = tf.reshape(destVec,[-1,2])[:,1]

        b = tf.reduce_sum(tf.multiply(SrcX,DestY) - tf.multiply(SrcY,DestX))
        b = b / Temp

        A = tf.reshape(tf.stack([a,b,-b,a]),[2,2])
        srcMean = tf.tensordot(srcMean,A,1)

        return tf.concat((tf.reshape(A,[-1]),destMean - srcMean),0)
    return tf.map_fn(lambda c: Do(c ,ShapeTo),ShapesFrom)
#test Good
Pixels = tf.constant(np.array([(x, y) for x in range(IMGSIZE) for y in range(IMGSIZE)], dtype=np.float32),shape=[IMGSIZE,IMGSIZE,2])
def AffineImage(Image,Transform,isInv=False):
    A = tf.reshape(Transform[:,0:4],[-1,2,2])
    T = tf.reshape(Transform[:,4:6],[-1,1,2])

    if isInv == False:
        A = tf.matrix_inverse(A)
        T = tf.matmul(-T,A)

    T = tf.reverse(T,[-1])
    A = tf.matrix_transpose(A)

    def Do(I,a,t):
        I = tf.reshape(I,[IMGSIZE,IMGSIZE])

        SrcPixels = tf.matmul(tf.reshape(Pixels,[IMGSIZE * IMGSIZE,2]),a) + t
        SrcPixels = tf.clip_by_value(SrcPixels,0,IMGSIZE - 1)

        outPixelsMinMin = tf.to_float(tf.to_int32(SrcPixels))
        dxdy = SrcPixels - outPixelsMinMin
        dx = dxdy[:,0]
        dy = dxdy[:,1]

        outPixelsMinMin = tf.reshape(tf.to_int32(outPixelsMinMin),[IMGSIZE * IMGSIZE,2])
        outPixelsMaxMin = tf.reshape(outPixelsMinMin + [1, 0],[IMGSIZE * IMGSIZE,2])
        outPixelsMinMax = tf.reshape(outPixelsMinMin + [0, 1],[IMGSIZE * IMGSIZE,2])
        outPixelsMaxMax = tf.reshape(outPixelsMinMin + [1, 1],[IMGSIZE * IMGSIZE,2])

        OutImage = (1 - dx) * (1 - dy) * tf.gather_nd(I,outPixelsMinMin) + dx * (1 - dy) * tf.gather_nd(I,outPixelsMaxMin) + (1 - dx) * dy * tf.gather_nd(I,outPixelsMinMax) + dx * dy * tf.gather_nd(I,outPixelsMaxMax)
        return tf.reshape(OutImage,[IMGSIZE,IMGSIZE,1])
    return tf.map_fn(lambda a:Do(a[0],a[1],a[2]),(Image,A,T),dtype=tf.float32)
#test Good
def AffineLandmark(Landmark, Transform,isInv=False):
    A = tf.reshape(Transform[:,0:4],[-1,2,2])
    T = tf.reshape(Transform[:,4:6],[-1,1,2])

    Landmark = tf.reshape(Landmark,[-1,LANDMARK,2])
    if isInv:
        A = tf.matrix_inverse(A)
        T = tf.matmul(-T,A)
    return tf.reshape(tf.matmul(Landmark,A) + T,[-1,LANDMARK * 2])
#test Good
def GetHeatMap(Landmark):
    HalfSize = 8
    def Do(L):
        def DoIn(Point):
            return Pixels - Point
        Landmarks = tf.reverse(tf.reshape(L,[-1,2]),[-1])
        Landmarks = tf.clip_by_value(Landmarks,HalfSize,112 - 1 - HalfSize)
        Ret = 1 / (tf.norm(tf.map_fn(DoIn,Landmarks),axis = 3) + 1)
        Ret = tf.reshape(tf.reduce_max(Ret,0),[IMGSIZE,IMGSIZE,1])
        return Ret
    return tf.map_fn(Do,Landmark)
#test Good
def PredictErr(GroudTruth,Predict):
    Gt = tf.reshape(GroudTruth,[-1,LANDMARK,2])
    Ot = tf.reshape(Predict,[-1,LANDMARK,2])

    def MeanErr(flt,Mix):
        Current,Gt = Mix
        MeanErr = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.subtract(Current, Gt) ** 2,1)))
        EyeDistance = tf.norm(tf.reduce_mean(Gt[LEFTEYE_START:LEFTEYE_END],0) - tf.reduce_mean(Gt[RIGHTEYE_START:RIGHTEYE_END],0))
        return MeanErr / EyeDistance

    return tf.scan(fn=MeanErr,elems=[Ot,Gt],initializer=0.0)

Feed_dict = {}
Ret_dict = {}

def Layers(Mshape=None):
    MeanShape = tf.constant(Mshape)

    with tf.variable_scope('Stage1'):
        InputImage = tf.placeholder(tf.float32,[None,IMGSIZE,IMGSIZE,1])
        GroundTruth = tf.placeholder(tf.float32,[None,LANDMARK * 2])
        S1_isTrain = tf.placeholder(tf.bool)

        Feed_dict['InputImage'] = InputImage
        Feed_dict['GroundTruth'] = GroundTruth
        Feed_dict['S1_isTrain'] = S1_isTrain

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

        S1_Fc1 = tf.layers.batch_normalization(tf.layers.dense(S1_DropOut,256,activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=S1_isTrain,name = 'S1_Fc1')
        S1_Fc2 = tf.layers.dense(S1_Fc1,LANDMARK * 2)

        S1_Ret = S1_Fc2 + MeanShape
        S1_Cost = tf.reduce_mean(PredictErr(GroundTruth,S1_Ret))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'Stage1')):
            S1_Optimizer = tf.train.AdamOptimizer(0.001).minimize(S1_Cost,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"Stage1"))
        
        Ret_dict['S1_Ret'] = S1_Ret
        Ret_dict['S1_Cost'] = S1_Cost
        Ret_dict['S1_Optimizer'] = S1_Optimizer

    with tf.variable_scope('Stage2'):
        S2_isTrain = tf.placeholder(tf.bool)
        Feed_dict['S2_isTrain'] = S2_isTrain

        S2_AffineParam = GetAffineParam(S1_Ret,MeanShape)
        S2_InputImage = AffineImage(InputImage,S2_AffineParam)
        S2_InputLandmark = AffineLandmark(S1_Ret,S2_AffineParam)
        S2_InputHeatmap = GetHeatMap(S2_InputLandmark)

        S2_Feature = tf.reshape(tf.layers.dense(S1_Fc1,int((IMGSIZE / 2) * (IMGSIZE / 2)),activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),(-1,int(IMGSIZE / 2),int(IMGSIZE / 2),1))
        S2_FeatureUpScale = tf.image.resize_images(S2_Feature,(IMGSIZE,IMGSIZE),1)

        S2_ConcatInput = tf.layers.batch_normalization(tf.concat([S2_InputImage,S2_InputHeatmap,S2_FeatureUpScale],3),training=S2_isTrain)
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
        S2_Fc2 = tf.layers.dense(S2_Fc1,LANDMARK * 2)

        S2_Ret = AffineLandmark(S2_Fc2 + S2_InputLandmark,S2_AffineParam,isInv=True)
        S2_Cost = tf.reduce_mean(PredictErr(GroundTruth,S2_Ret))

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'Stage2')):
            S2_Optimizer = tf.train.AdamOptimizer(0.001).minimize(S2_Cost,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"Stage2"))

        Ret_dict['S2_Ret'] = S2_Ret
        Ret_dict['S2_Cost'] = S2_Cost
        Ret_dict['S2_Optimizer'] = S2_Optimizer

        Ret_dict['S2_InputImage'] = S2_InputImage
        Ret_dict['S2_InputLandmark'] = S2_InputLandmark
        Ret_dict['S2_InputHeatmap'] = S2_InputHeatmap
        Ret_dict['S2_FeatureUpScale'] = S2_FeatureUpScale
    return

def JeloTest(Shape,Sess):

    def transform(form,to):
        destMean = np.mean(to, axis=0)
        srcMean = np.mean(form, axis=0)

        srcVec = (form - srcMean).flatten()
        destVec = (to - destMean).flatten()

        a = np.dot(srcVec, destVec) / np.linalg.norm(srcVec) ** 2
        b = 0
        for i in range(form.shape[0]):
            b += srcVec[2 * i] * destVec[2 * i + 1] - srcVec[2 * i + 1] * destVec[2 * i] 
        b = b / np.linalg.norm(srcVec) ** 2

        T = np.array([[a, b], [-b, a]])
        srcMean = np.dot(srcMean, T)

        return T, destMean - srcMean

    initLandmark = np.array([[142.276932,298.257019],
        [198.306290,300.303680],
        [204.209900,321.991486],
        [128.933411,326.152954],
        [286.840668,323.690918],
        [296.065277,302.031494],
        [347.137054,309.349060],
        [352.773895,334.986145],
        [144.340683,362.945618],
        [171.983276,347.863495],
        [201.002991,359.234192],
        [171.996918,368.796600],
        [285.312592,363.825684],
        [316.107147,355.429962],
        [342.870148,372.732086],
        [313.159668,376.430511],
        [188.402573,496.530182],
        [200.569229,482.153229],
        [221.423798,475.474884],
        [238.420578,479.840332],
        [255.591629,477.686157],
        [273.992218,486.573059],
        [286.708618,502.069000],
        [273.553467,516.843567],
        [256.615479,526.680115],
        [237.759949,527.476990],
        [219.077271,524.133423],
        [200.404175,513.671265],
        [201.942215,496.860443],
        [217.904770,494.473602],
        [238.952606,494.496460],
        [262.185150,496.672821],
        [275.234283,501.618683],
        [261.826233,502.228607],
        [240.554810,501.575562],
        [218.696976,499.738373]],dtype = np.float32)
    Shape = Shape.reshape(LANDMARK,2)

    Device = cv2.VideoCapture('D:\\work\\FaceAlignment\\CNN_DAN\\DAN\\DAN\\Jelo\\zhao.mp4')
    while True:
        _,Frame = Device.read()
        FrameGray = cv2.cvtColor(Frame,cv2.COLOR_BGR2GRAY)

        A,T = transform(initLandmark,Shape)
        A = np.linalg.inv(A)
        T = np.dot(-T, A)
        Img = ndimage.interpolation.affine_transform(FrameGray,A,T[[1,0]],output_shape=(IMGSIZE,IMGSIZE))
        Img = np.reshape(Img,(IMGSIZE,IMGSIZE,1)).astype(np.float32) / 255.0
        RetLandmark = Sess.run(Ret_dict['S1_Ret'],{Feed_dict['InputImage']:[Img],Feed_dict['S1_isTrain']:False,Feed_dict['S2_isTrain']:False})[0].reshape(LANDMARK,2)

        initLandmark = np.dot(RetLandmark,A) + T
        
        for i in range(LANDMARK):
            cv2.circle(Frame,(int(initLandmark[i,0]),int(initLandmark[i,1])),3,(0,255,0),-1)

        cv2.imshow('JeloTest',Frame)
        cv2.waitKey(15)


def Landmark68Test(Shape,ImgMean,ImgStd,Sess):

    def transform(form,to):
        destMean = np.mean(to, axis=0)
        srcMean = np.mean(form, axis=0)

        srcVec = (form - srcMean).flatten()
        destVec = (to - destMean).flatten()

        a = np.dot(srcVec, destVec) / np.linalg.norm(srcVec) ** 2
        b = 0
        for i in range(form.shape[0]):
            b += srcVec[2 * i] * destVec[2 * i + 1] - srcVec[2 * i + 1] * destVec[2 * i] 
        b = b / np.linalg.norm(srcVec) ** 2

        T = np.array([[a, b], [-b, a]])
        srcMean = np.dot(srcMean, T)

        return T, destMean - srcMean

    Shape = Shape.reshape(LANDMARK,2)

    Device = cv2.VideoCapture(0)
    # VideoWriter = cv2.VideoWriter('OutMy.avi',cv2.VideoWriter_fourcc(*'XVID'),30,(640,480))
    Cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    Rest = True
    initLandmark = None

    while True:
        _,Frame = Device.read()
        FrameGray = cv2.cvtColor(Frame,cv2.COLOR_BGR2GRAY)

        if Rest:
            Rects = Cascade.detectMultiScale(FrameGray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))
            if len(Rects) > 0:
                #print( [Rects[0][0],[Rects[0][1]]])
                #print(Shape / IMGSIZE * min(Rects[0][2],Rects[0][3]))
                initLandmark = Shape / IMGSIZE * min(Rects[0][2],Rects[0][3]) + [Rects[0][0],Rects[0][1]]
                A,T = transform(initLandmark,Shape)
                A = np.linalg.inv(A)
                T = np.dot(-T, A)
                Img = ndimage.interpolation.affine_transform(FrameGray,A,T[[1,0]],output_shape=(IMGSIZE,IMGSIZE))
                Img = np.reshape(Img,(IMGSIZE,IMGSIZE,1)).astype(np.float32) / 255.0
                Img = (Img - ImgMean) / ImgStd
                RetLandmark = Sess.run(Ret_dict['S2_Ret'],{Feed_dict['InputImage']:[Img],Feed_dict['S1_isTrain']:False,Feed_dict['S2_isTrain']:False})[0].reshape(LANDMARK,2)
                initLandmark = np.dot(RetLandmark,A) + T
                initLandmark = np.round(initLandmark).astype(np.int32)
                for i in range(LANDMARK):
                    cv2.circle(Frame,(initLandmark[i,0],initLandmark[i,1]),2,(0,255,0),-1)
                Rest = False
        else:
            A,T = transform(initLandmark,Shape)
            A = np.linalg.inv(A)
            T = np.dot(-T, A)
            Img = ndimage.interpolation.affine_transform(FrameGray,A,T[[1,0]],output_shape=(IMGSIZE,IMGSIZE))
            Img = np.reshape(Img,(IMGSIZE,IMGSIZE,1)).astype(np.float32) / 255.0
            Img = (Img - ImgMean) / ImgStd
            RetLandmark = Sess.run(Ret_dict['S2_Ret'],{Feed_dict['InputImage']:[Img],Feed_dict['S1_isTrain']:False,Feed_dict['S2_isTrain']:False})[0].reshape(LANDMARK,2)
            initLandmark = np.dot(RetLandmark,A) + T
            initLandmark = np.round(initLandmark).astype(np.int32)
            for i in range(LANDMARK):
                cv2.circle(Frame,(initLandmark[i,0],initLandmark[i,1]),2,(0,255,0),-1)


        cv2.imshow('Test',Frame)
        if cv2.waitKey(1) == 32:
            Rest = True
        # VideoWriter.write(Frame)

_300W = np.load('AllDataset.npz')
I = _300W['Image']
G = _300W['Landmark']
MeanShape = _300W['MeanShape']
ImageMean = _300W['ImgMean']
ImageStd = _300W['ImgStd']

_300WTest = np.load('300WTestCommon.npz')
Ti = _300WTest['Image']
Tg = _300WTest['Landmark']


#for i in range(500):
#    for k in range(68):
#        cv2.circle(I[i],(int(G[i][k * 2]),int(G[i][k * 2 + 1])),1,(255))
#        cv2.circle(Ti[i],(int(Tg[i][k * 2]),int(Tg[i][k * 2 + 1])),1,(255))

#    cv2.imshow('TrainSrc',I[i])
#    cv2.imshow('TestSrc',Ti[i])
#    cv2.waitKey(-1)

Layers(MeanShape)
STAGE = 2

with tf.Session() as Sess:
    Saver = tf.train.Saver()
    Writer = tf.summary.FileWriter("logs/", Sess.graph)
    if STAGE == 0:
        Sess.run(tf.global_variables_initializer())
    else:
        Saver.restore(Sess,'./Model/Model')
        print('Model Read Over!')
       
    Landmark68Test(MeanShape,ImageMean,ImageStd,Sess)

    for w in range(1000):
        Count = 0
        while Count * 64 < I.shape[0]  :
            RandomIdx = np.random.choice(I.shape[0],64,False)
            if STAGE == 1 or STAGE == 0:
                Sess.run(Ret_dict['S1_Optimizer'],{Feed_dict['InputImage']:I[RandomIdx],Feed_dict['GroundTruth']:G[RandomIdx],Feed_dict['S1_isTrain']:True,Feed_dict['S2_isTrain']:False})
            else:
                Sess.run(Ret_dict['S2_Optimizer'],{Feed_dict['InputImage']:I[RandomIdx],Feed_dict['GroundTruth']:G[RandomIdx],Feed_dict['S1_isTrain']:False,Feed_dict['S2_isTrain']:True})

            if Count % 256 == 0:
                TestErr = 0
                BatchErr = 0

                if STAGE == 1 or STAGE == 0:
                    TestErr = Sess.run(Ret_dict['S1_Cost'],{Feed_dict['InputImage']:Ti,Feed_dict['GroundTruth']:Tg,Feed_dict['S1_isTrain']:False,Feed_dict['S2_isTrain']:False})
                    BatchErr = Sess.run(Ret_dict['S1_Cost'],{Feed_dict['InputImage']:I[RandomIdx],Feed_dict['GroundTruth']:G[RandomIdx],Feed_dict['S1_isTrain']:False,Feed_dict['S2_isTrain']:False})
                else:
                    #Landmark,Img,HeatMap,FeatureUpScale =
                    #Sess.run([Ret_dict['S2_InputLandmark'],Ret_dict['S2_InputImage'],Ret_dict['S2_InputHeatmap'],Ret_dict['S2_FeatureUpScale']],{Feed_dict['InputImage']:I[RandomIdx],Feed_dict['GroundTruth']:G[RandomIdx],Feed_dict['S1_isTrain']:False,Feed_dict['S2_isTrain']:False})
                    #for i in range(64):
                    #    TestImage = np.zeros([112,112,1])
                    #    for p in range(68):
                    #        cv2.circle(TestImage,(int(Landmark[i][p *
                    #        2]),int(Landmark[i][p * 2 + 1])),1,(255),-1)

                    #    cv2.imshow('Landmark',TestImage)
                    #    cv2.imshow('Image',Img[i])
                    #    cv2.imshow('HeatMap',HeatMap[i])
                    #    cv2.imshow('FeatureUpScale',FeatureUpScale[i])
                    #    cv2.waitKey(-1)
                    TestErr = Sess.run(Ret_dict['S2_Cost'],{Feed_dict['InputImage']:Ti,Feed_dict['GroundTruth']:Tg,Feed_dict['S1_isTrain']:False,Feed_dict['S2_isTrain']:False})
                    BatchErr = Sess.run(Ret_dict['S2_Cost'],{Feed_dict['InputImage']:I[RandomIdx],Feed_dict['GroundTruth']:G[RandomIdx],Feed_dict['S1_isTrain']:False,Feed_dict['S2_isTrain']:False})
                print(w,Count,'TestErr:',TestErr,' BatchErr:',BatchErr)
            Count += 1
        Saver.save(Sess,'./Model/Model')
