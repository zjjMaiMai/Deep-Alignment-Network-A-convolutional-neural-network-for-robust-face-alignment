import tensorflow as tf
import numpy as np  
import AffineTransform as Af
import itertools

class FaceAlignment(object):
    def __init__(self):
        self.Feed_dict = {}
        self.Ret_dict = {}

        self.LandmarkNum = 0
        self.LandmarkPatchSize = 16
        self.InitLandmark = None

        self.ImageSize = 0

        self.Image = []
        self.Gt = []
     
    def BuildLayer(self):
        with tf.variable_scope('Stage1'):
            self.Feed_dict['S1_InputImage'] = tf.placeholder(tf.float32,[None,112,112,1])
            self.Feed_dict['S1_GroundTruth'] = tf.placeholder(tf.float32,[None,136])
            self.Feed_dict['S1_InputLandmark'] = tf.placeholder(tf.float32,[136])
            self.Feed_dict['S1_isTrain'] = tf.placeholder(tf.bool)

            S1_Conv1a = tf.layers.batch_normalization(tf.layers.conv2d(self.Feed_dict['S1_InputImage'],64,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=self.Feed_dict['S1_isTrain'])
            S1_Conv1b = tf.layers.batch_normalization(tf.layers.conv2d(S1_Conv1a,64,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=self.Feed_dict['S1_isTrain'])
            S1_Pool1 = tf.layers.max_pooling2d(S1_Conv1b,2,2,padding='same')

            S1_Conv2a = tf.layers.batch_normalization(tf.layers.conv2d(S1_Pool1,128,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=self.Feed_dict['S1_isTrain'])
            S1_Conv2b = tf.layers.batch_normalization(tf.layers.conv2d(S1_Conv2a,128,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=self.Feed_dict['S1_isTrain'])
            S1_Pool2 = tf.layers.max_pooling2d(S1_Conv2b,2,2,padding='same')

            S1_Conv3a = tf.layers.batch_normalization(tf.layers.conv2d(S1_Pool2,256,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=self.Feed_dict['S1_isTrain'])
            S1_Conv3b = tf.layers.batch_normalization(tf.layers.conv2d(S1_Conv3a,256,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=self.Feed_dict['S1_isTrain'])
            S1_Pool3 = tf.layers.max_pooling2d(S1_Conv3b,2,2,padding='same')

            S1_Conv4a = tf.layers.batch_normalization(tf.layers.conv2d(S1_Pool3,512,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=self.Feed_dict['S1_isTrain'])
            S1_Conv4b = tf.layers.batch_normalization(tf.layers.conv2d(S1_Conv4a,512,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=self.Feed_dict['S1_isTrain'])
            S1_Pool4 = tf.layers.max_pooling2d(S1_Conv4b,2,2,padding='same')

            S1_Pool4_Flat = tf.contrib.layers.flatten(S1_Pool4)
            S1_DropOut = tf.layers.dropout(S1_Pool4_Flat,0.5,training=self.Feed_dict['S1_isTrain'])

            S1_Fc1 = tf.layers.batch_normalization(tf.layers.dense(S1_DropOut,256,activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=self.Feed_dict['S1_isTrain'])
            S1_Fc2 = tf.layers.dense(S1_Fc1,136)

            S1_Ret = S1_Fc2 + self.Feed_dict['S1_InputLandmark']
            S1_Cost = tf.reduce_mean(self.PredictErr(self.Feed_dict['S1_GroundTruth'],S1_Ret))

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'Stage1')):
                S1_Optimizer = tf.train.AdamOptimizer(0.001).minimize(S1_Cost,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"Stage1"))
        
            self.Ret_dict['S1_Ret'] = S1_Ret
            self.Ret_dict['S1_Cost'] = S1_Cost
            self.Ret_dict['S1_Optimizer'] = S1_Optimizer

        with tf.variable_scope('Stage2'):
            self.Feed_dict['S2_InputImage'] = tf.placeholder(tf.float32,[None,112,112,1])
            self.Feed_dict['S2_InputHeatMap'] = tf.placeholder(tf.float32,[None,112,112,1])
            self.Feed_dict['S2_InputLandmark'] = tf.placeholder(tf.float32,[None,136])
            self.Feed_dict['S2_GroundTruth'] = tf.placeholder(tf.float32,[None,136])
            self.Feed_dict['S2_isTrain'] = tf.placeholder(tf.bool)

            S2_Feature = tf.reshape(tf.layers.dense(S1_Fc1,56 * 56,activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),(-1,56,56,1))
            S2_FeatureUpScale = tf.image.resize_images(S2_Feature,(112,112),1)

            S2_ConcatInput = tf.layers.batch_normalization(tf.concat([self.Feed_dict['S2_InputImage'],self.Feed_dict['S2_InputHeatMap'],S2_FeatureUpScale],3),training=self.Feed_dict['S2_isTrain'])
            S2_Conv1a = tf.layers.batch_normalization(tf.layers.conv2d(S2_ConcatInput,64,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=self.Feed_dict['S2_isTrain'])
            S2_Conv1b = tf.layers.batch_normalization(tf.layers.conv2d(S2_Conv1a,64,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=self.Feed_dict['S2_isTrain'])
            S2_Pool1 = tf.layers.max_pooling2d(S2_Conv1b,2,2,padding='same')

            S2_Conv2a = tf.layers.batch_normalization(tf.layers.conv2d(S2_Pool1,128,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=self.Feed_dict['S2_isTrain'])
            S2_Conv2b = tf.layers.batch_normalization(tf.layers.conv2d(S2_Conv2a,128,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=self.Feed_dict['S2_isTrain'])
            S2_Pool2 = tf.layers.max_pooling2d(S2_Conv2b,2,2,padding='same')

            S2_Conv3a = tf.layers.batch_normalization(tf.layers.conv2d(S2_Pool2,256,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=self.Feed_dict['S2_isTrain'])
            S2_Conv3b = tf.layers.batch_normalization(tf.layers.conv2d(S2_Conv3a,256,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=self.Feed_dict['S2_isTrain'])
            S2_Pool3 = tf.layers.max_pooling2d(S2_Conv3b,2,2,padding='same')

            S2_Conv4a = tf.layers.batch_normalization(tf.layers.conv2d(S2_Pool3,512,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=self.Feed_dict['S2_isTrain'])
            S2_Conv4b = tf.layers.batch_normalization(tf.layers.conv2d(S2_Conv4a,512,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=self.Feed_dict['S2_isTrain'])
            S2_Pool4 = tf.layers.max_pooling2d(S2_Conv4b,2,2,padding='same')

            S2_Pool4_Flat = tf.contrib.layers.flatten(S2_Pool4)
            S2_DropOut = tf.layers.dropout(S2_Pool4_Flat,0.5,training=self.Feed_dict['S2_isTrain'])

            S2_Fc1 = tf.layers.batch_normalization(tf.layers.dense(S2_DropOut,256,activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()),training=self.Feed_dict['S2_isTrain'])
            S2_Fc2 = tf.layers.dense(S2_Fc1,136)

            S2_Ret = S2_Fc2 + self.Feed_dict['S2_InputLandmark']
            S2_Cost = tf.reduce_mean(self.PredictErr(self.Feed_dict['S2_GroundTruth'],S2_Ret))

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS,'Stage2')):
                S2_Optimizer = tf.train.AdamOptimizer(0.001).minimize(S2_Cost,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,"Stage2"))

            self.Ret_dict['S2_Ret'] = S2_Ret
            self.Ret_dict['S2_Cost'] = S2_Cost
            self.Ret_dict['S2_Optimizer'] = S2_Optimizer

    def PredictErr(self,GroudTruth,Predict):
        Gt = tf.reshape(GroudTruth,[-1,self.LandmarkNum,2])
        Ot = tf.reshape(Predict,[-1,self.LandmarkNum,2])

        def MeanErr(flt,Mix):
            Current,Gt = Mix
            MeanErr = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.subtract(Current, Gt) ** 2,1)))
            EyeDistance = tf.norm(tf.reduce_mean(Gt[36:42],0) - tf.reduce_mean(Gt[42:48],0))
            return MeanErr / EyeDistance

        return tf.scan(fn=MeanErr,elems=[Ot,Gt],initializer=0.0)

    def GetAffineParam(ShapesFrom,ShapeTo):
        destination = tf.reshape(ShapeTo,[-1,2])
        destMean = tf.reduce_mean(destination,0)
        destCenter = destination - destMean
        destVec = tf.reshape(destCenter,[-1])
        DestX = tf.reshape(destVec,[-1,2])[:,0]
        DestY = tf.reshape(destVec,[-1,2])[:,1]

        def Do(From):
            source = tf.reshape(From,[-1,2])
            srcMean = tf.reduce_mean(source,0)

            srcCenter = source - srcMean
            srcVec = tf.reshape(srcCenter,[-1])

            Temp = tf.norm(srcVec) ** 2
            a = tf.tensordot(srcVec,destVec,1) / Temp
            b = 0

            SrcX = tf.reshape(srcVec,[-1,2])[:,0]
            SrcY = tf.reshape(srcVec,[-1,2])[:,1]

            b = tf.reduce_sum(tf.multiply(SrcX,DestY) - tf.multiply(SrcY,DestX))
            b = b / Temp

            A = tf.reshape(tf.stack([a,b,-b,a]),[2,2])
            srcMean = tf.tensordot(srcMean,A,1)

            return tf.concat((tf.reshape(A,[-1]),destMean - srcMean),0)

        return tf.scan(Do,ShapesFrom,initializer=tf.zeros_initializer())

    def AffineImage(Image,Transform,isInv=False):
        A = tf.reshape(Transform[:,0:4],[-1,2,2])
        T = tf.reshape(Transform[:,4:6],[-1,1,2])

        if isInv == False:
            A = tf.matrix_inverse(A)
            T = tf.matmul(-T,A)

        def Do(I,a,t):
            I = tf.reshape(I,[112,112])

            t = tf.gather_nd(t,[[0,1],[0,0]])
            a = tf.transpose(a)

            SrcPixels = tf.matmul(Pixels,a) + t
            SrcPixels = tf.clip_by_value(SrcPixels,0,112 - 1)

            SrcPixelsIdx = tf.reshape(tf.to_int32(SrcPixels),[112 * 112,2])
            OutImage = tf.gather_nd(I,SrcPixelsIdx)
            return tf.reshape(OutImage,[112,112,1])
        return tf.scan(Do,elems=[Image,A,T],initializer=tf.zeros_initializer())

    def TrainStage1(self,e,b):
        with tf.Session() as Sess:
            Saver = tf.train.Saver()
            try:
                Saver.restore(Sess,'./Model1/Model')
            except:
                Sess.run(tf.global_variables_initializer())

            for Epoch in range(e):
                for BatchCount in range(self.Image.shape[0] // b):
                    RandomIdx = np.random.choice(self.Image.shape[0],b,False)
                    Sess.run(self.Ret_dict['S1_Optimizer'],
                             {self.Feed_dict['S1_InputImage']:self.Image[RandomIdx],
                              self.Feed_dict['S1_InputLandmark']:self.InitLandmark,
                              self.Feed_dict['S1_GroundTruth']:self.Gt[RandomIdx],
                              self.Feed_dict['S1_isTrain']:True})

                print('Epoch : ',Epoch,'Test Cost:',Sess.run(self.Ret_dict['S1_Cost'],
                                                             {self.Feed_dict['S1_InputImage']:self.TestImage,
                                                              self.Feed_dict['S1_InputLandmark']:self.InitLandmark,
                                                              self.Feed_dict['S1_GroundTruth']:self.TestGt,
                                                              self.Feed_dict['S1_isTrain']:False}))
                Saver.save(Sess,'./Model1/Model')
        return

    def TrainStage2(self,e,b):
        with tf.Session() as Sess:
            Saver = tf.train.Saver()
            Saver.restore(Sess,'./Model1/Model')

            #GetLastTempOutput
            S2_InputImage = np.array()
            S2_InputHeatMap = np.array()
            S2_InputLandmark = np.array()
            S2_GroundTruth = np.array()
            Transform = np.array()

            StartIdx = 0

            while StartIdx < self.Image.shape[0]:
                EndIdx = StartIdx + 64 if StartIdx + 64 < self.Image.shape[0] else self.Image.shape[0]
                S1_OutLandmark = Sess.run(self.Ret_dict['S1_Ret'],
                                          {self.Feed_dict['S1_InputImage']:self.Image[StartIdx:EndIdx],
                                           self.Feed_dict['S1_InputLandmark']:self.InitLandmark,
                                           self.Feed_dict['S1_isTrain']:False})

                TransformParam = self.GetAffineParam(S1_OutLandmark)
                S2_InputLandmark = np.append(S2_InputLandmark,AffineLandmark(S1_OutLandmark,TransformParam),0)
                S2_GroundTruth = np.append(S2_GroundTruth,AffineLandmark(self.Gt[StartIdx:EndIdx],TransformParam),0)
                S2_InputImage = np.append(S2_InputImage,AffineImage(self.Image[StartIdx:EndIdx],TransformParam),0)
                S2_InputHeatMap = np.append(S2_InputHeatMap,GetHeatMap(S2_InputLandmark),0)
                Transform = np.append(Transform,TransformParam,0)

            for Epoch in range(e):
                for BatchCount in range(self.Image.shape[0] // b):
                    RandomIdx = np.random.choice(self.Image.shape[0],b,False)
                    Sess.run(self.Ret_dict['S2_Optimizer'],
                             {self.Feed_dict['S2_InputImage']:S2_InputImage[RandomIdx],
                              self.Feed_dict['S2_InputLandmark']:S2_InputLandmark[RandomIdx],
                              self.Feed_dict['S2_GroundTruth']:S2_GroundTruth[RandomIdx],
                              self.Feed_dict['S2_InputHeatMap']:S2_InputHeatMap[RandomIdx],
                              self.Feed_dict['S2_isTrain']:True})

                print('Epoch : ',Epoch,'Test Cost:',Sess.run(self.Ret_dict['S2_Cost'],
                                                             {self.Feed_dict['S1_InputImage']:self.TestImage,
                                                              self.Feed_dict['S1_InputLandmark']:self.InitLandmark,
                                                              self.Feed_dict['S1_GroundTruth']:self.TestGt,
                                                              
                                                              self.Feed_dict['S1_isTrain']:False}))
                Saver.save(Sess,'./Model1/Model')
        return

                