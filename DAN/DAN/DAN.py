import tensorflow as tf
import numpy as np  
import matplotlib.pyplot as plt  
import cv2
import pickle
import matplotlib.pyplot as plt


DropOutRate = 0.5
isTrain = True

def bn(x):
    return tf.layers.batch_normalization(x,training=isTrain)
    #return x
def show_landmarks(image, landmarks):
    """Show image with landmarks"""
    landmarks = landmarks * 112
    image = np.reshape(image,[112,112])
    plt.imshow(image)
    landmarks = np.reshape(landmarks,[68,2])
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.',c ='w')
    plt.pause(1.500)  # pause a bit so that plots are updated
    plt.close()

def PredictErr(GroudTruth,Predict):
    Gt = tf.reshape(GroudTruth,[-1,68,2])
    Ot = tf.reshape(Predict,[-1,68,2])

    def MeanErr(flt,Mix):
        Current,Gt = Mix
        MeanErr = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.subtract(Current, Gt) ** 2,1)))
        EyeDistance = tf.norm(tf.reduce_mean(Gt[36:42],0) - tf.reduce_mean(Gt[42:48],0))
        return MeanErr / EyeDistance

    return tf.scan(fn=MeanErr,elems=[Ot,Gt],initializer=0.0)

def ReadFormImageList(Path):
    List = open(Path)
    ImagePathList = List.readlines()

    Gts = []
    Images = [] 

    for ImagePath in ImagePathList:
        Image = cv2.imread(ImagePath.strip(),cv2.IMREAD_GRAYSCALE)

        PtsPath = ImagePath[:-4] + 'pts'
        Pts = open(PtsPath)
        Data = Pts.readlines()[3:-1]

        NumData = np.empty((68,2)).astype(np.float32)

        for i in range(68):
            Points = Data[i].split()
            NumData[i][0] = float(Points[0])
            NumData[i][1] = float(Points[1])

        Max = NumData.max(0).astype(np.integer)
        Min = NumData.min(0).astype(np.integer)

        Image = Image[Min[1]:Max[1],Min[0]:Max[0]]
        for i in range(68):
            NumData[i][0] = (NumData[i][0] - Min[0]) / (Max[0] - Min[0])
            NumData[i][1] = (NumData[i][1] - Min[1]) / (Max[1] - Min[1])

        Image = np.reshape(cv2.resize(Image,(112,112)),(112,112,1))

        Gts.append(NumData)
        Images.append(Image)
    return Images,Gts

def Stage1(InputS0=None):
    InputImage = tf.placeholder(tf.float32,[None,112,112,1])
    GroudTruth = tf.placeholder(tf.float32,[None,136])

    S0 = tf.constant(InputS0)

    Conv1a = bn(tf.layers.conv2d(tf.cast(InputImage,tf.float32),64,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()))
    Conv1b = bn(tf.layers.conv2d(Conv1a,64,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()))
    Pool1 = tf.layers.max_pooling2d(Conv1b,2,2,padding='same')

    Conv2a = bn(tf.layers.conv2d(Pool1,128,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()))
    Conv2b = bn(tf.layers.conv2d(Conv2a,128,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()))
    Pool2 = tf.layers.max_pooling2d(Conv2b,2,2,padding='same')

    Conv3a = bn(tf.layers.conv2d(Pool2,256,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()))
    Conv3b = bn(tf.layers.conv2d(Conv3a,256,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()))
    Pool3 = tf.layers.max_pooling2d(Conv3b,2,2,padding='same')

    Conv4a = bn(tf.layers.conv2d(Pool3,512,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()))
    Conv4b = bn(tf.layers.conv2d(Conv4a,512,3,1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()))
    Pool4 = tf.layers.max_pooling2d(Conv4b,2,2,padding='same')

    Pool4_Flat = tf.contrib.layers.flatten(Pool4)
    DropOut = tf.layers.dropout(Pool4_Flat,DropOutRate)

    Fc1 = bn(tf.layers.dense(DropOut,256,activation=tf.nn.relu,kernel_initializer=tf.glorot_uniform_initializer()))
    Fc2 = tf.layers.dense(Fc1,136)

    Ret = Fc2 + S0

    Cost = tf.reduce_mean(PredictErr(GroudTruth,Ret))
    #Cost = tf.reduce_mean(tf.square(Ret - GroudTruth)) / 2.0)
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        Optimizer = tf.train.AdamOptimizer(1e-3).minimize(Cost)

    return InputImage,GroudTruth,Optimizer,Ret,Cost,Fc2




File = open("ImageData.pkl","rb")
I = pickle.load(File)
G = pickle.load(File)
G = np.reshape(G,[-1,136])
MeanShape = np.mean(G,0)

MeanShape = np.where(True,MeanShape,0)

InputImage,GroudTruth,Optimizer,Ret,Cost,Fc2 = Stage1(MeanShape)

with tf.Session() as Sess:
    Sess.run(tf.global_variables_initializer())
    Count = 0
    while True:
        startIdx = np.random.randint(0,len(I) - 64)
        endIdx = startIdx + 64
        #startIdx = 0
        #endIdx = 64
        Sess.run(Optimizer,{InputImage:I[startIdx:endIdx],GroudTruth:G[startIdx:endIdx]})
        Count += 1
        if Count % 100 == 0:
            isTrain = False
            c = Sess.run(Cost,{InputImage:I[startIdx:endIdx],GroudTruth:G[startIdx:endIdx]})
            print(c)
            isTrain = True