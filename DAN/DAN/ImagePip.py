import numpy as np
import cv2
import math
#读图片
'''
Data augmentation is performed by mirroring around the Y axis as well as 
random translation, rotation and scaling, all sampled from normal distributions. 
During data augmentation a total of 10 images are created from each input image in the training set
'''

    

def Load(fileList):
    ImgList = []
    PtsList = []
    ImgPathList,PtsPathList = LoadImageList(fileList)
    for Imgfile,Ptsfile in zip(ImgPathList,PtsPathList):
        Img = LoadOneImage(Imgfile)
        Pts = LoadOnePts(Ptsfile)

        MeanPoints = np.mean(Pts,axis=0)
        Max = np.max(np.abs(Pts - MeanPoints)) * 1.5


        Img = Img[int(MeanPoints[1] - Max) : int(math.ceil(MeanPoints[1] + Max)),int(MeanPoints[0] - Max) : int(math.ceil(MeanPoints[0] + Max))]
        cv2.imshow('T',Img)
        cv2.waitKey(-1)

        #cv2.

    
def LoadImageList(file):
    ImgList = []
    PtsList = []
    with open(file) as f:
        line = f.readline()
        while line:
            ImgList.append(line[:-1])
            PtsList.append(line[:-5] + '.pts')
            line = f.readline()
    return ImgList,PtsList

def LoadOnePts(ptsfile):
    Strary = []
    with open(ptsfile) as f:
        line = f.readline()
        line = f.readline()
        NumPoints = int(line.rsplit(' ',1)[-1])
        line = f.readline()
        for i in range(NumPoints):
            line = f.readline() 
            Strary.append(list(map(float,line.split(' '))))
    return np.array(Strary)

def LoadOneImage(imgfile):
    img = cv2.imread(imgfile,cv2.IMREAD_GRAYSCALE)
    return img        

Load('D:\\Dataset\\PaperTest.txt')