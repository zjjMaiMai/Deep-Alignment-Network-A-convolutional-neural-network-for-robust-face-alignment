import numpy as np
import cv2
import math
import pickle
#读图片
'''
Data augmentation is performed by mirroring around the Y axis as well as 
random translation, rotation and scaling, all sampled from normal distributions. 
During data augmentation a total of 10 images are created from each input image in the training set
'''

def Load(fileList,isTrainData=True):
    Count = 0
    ImgList = []
    PtsList = []
    ImgPathList,PtsPathList = LoadImageList(fileList)
    for Imgfile,Ptsfile in zip(ImgPathList,PtsPathList):
        Img = LoadOneImage(Imgfile)
        Pts = LoadOnePts(Ptsfile)
        Count += 1

        h,w = np.shape(Img)
        if (h > 800 or w > 800):
            Img = cv2.resize(Img,(int(w / 4.0),int(h / 4.0)))
            Pts = Pts / 4.0
        MeanPoints = np.mean(Pts,axis=0)
        Max = np.max(np.abs(Pts - MeanPoints)) * 1.4

        left = int(MeanPoints[0] - Max)
        right = int(math.ceil(MeanPoints[0] + Max))
        top = int(MeanPoints[1] - Max)
        down = int(math.ceil(MeanPoints[1] + Max))

        ImgR = Img[0 if top < 0 else top:down if down < h else h,0 if left < 0 else left:right if right < w else w]
        
        if left < 0:
            for i in range(abs(left)):
                ImgR = np.insert(ImgR,0,0,axis=1)
        if top < 0:
            for i in range(abs(top)):
                ImgR = np.insert(ImgR,0,0,axis=0)
        if right > w:
            for i in range(right - w):
                ImgR = np.insert(ImgR,np.shape(ImgR)[1],0,axis=1)
        if down > h:
            for i in range(down - h):
                ImgR = np.insert(ImgR,np.shape(ImgR)[0],0,axis=0)

        ImgR = cv2.resize(ImgR,(112,112))
        w = right - left
        h = down - top
        PtsR = Pts.copy()
        for i in range(len(Pts)):
            PtsR[i] = [(Pts[i][0] - left) / w ,(Pts[i][1] - top) / h]

        #for i in range(68):
        #    cv2.circle(ImgR,(int(PtsR[i][0]),int(PtsR[i][1])),2,(255),-1)
        #cv2.imshow('s',ImgR)
        #cv2.waitKey(-1)

        ImgList.append(ImgR)
        PtsList.append(PtsR)
        if isTrainData:
            for DaNum in range(9):
                Scale = np.random.normal(1.5,0.2)
                Max = np.max(np.abs(Pts - MeanPoints)) * Scale

                h,w = np.shape(Img)
            
                RandomDx = np.random.normal(0.0,0.1) 
                RandomDy = np.random.normal(0.0,0.1) 

                left = int(MeanPoints[0] - Max * (1 + RandomDx))
                right = int(math.ceil(MeanPoints[0] + Max * (1 + RandomDx)))
                top = int(MeanPoints[1] - Max * (1 + RandomDy))
                down = int(math.ceil(MeanPoints[1] + Max * (1 + RandomDy)))

                ImgR = Img[0 if top < 0 else top:down if down < h else h,0 if left < 0 else left:right if right < w else w]
        
                if left < 0:
                    for i in range(abs(left)):
                        ImgR = np.insert(ImgR,0,0,axis=1)
                if top < 0:
                    for i in range(abs(top)):
                        ImgR = np.insert(ImgR,0,0,axis=0)
                if right > w:
                    for i in range(right - w):
                        ImgR = np.insert(ImgR,np.shape(ImgR)[1],0,axis=1)
                if down > h:
                    for i in range(down - h):
                        ImgR = np.insert(ImgR,np.shape(ImgR)[0],0,axis=0)

                ImgR = cv2.resize(ImgR,(112,112))
                w = right - left
                h = down - top
                PtsR = Pts.copy()
                for i in range(len(PtsR)):
                    PtsR[i] = [(Pts[i][0] - left) / w * 112,(Pts[i][1] - top) / h * 112]

                RandomRotateM = cv2.getRotationMatrix2D((112 / 2,112 / 2),np.random.normal(0,15),1.)
                ImgR = cv2.warpAffine(ImgR,RandomRotateM,(112,112))

                LandmarkNum = len(PtsR)
                PtsR = np.reshape(PtsR,(LandmarkNum,1,2))
                PtsR = cv2.transform(PtsR,RandomRotateM)
                PtsR = np.reshape(PtsR,(LandmarkNum,2)) / 112.0
            
                #for i in range(68):
                #    cv2.circle(ImgR,(int(PtsR[i][0]),int(PtsR[i][1])),2,(255),-1)
                #cv2.imshow('s',ImgR)
                #cv2.waitKey(-1)
                print('[',Count,':',DaNum,']')

                ImgList.append(ImgR)
                PtsList.append(PtsR)
    return ImgList,PtsList
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
            line = f.readline().strip('\n').split()
            Strary.append(list(map(float,line)))
    return np.array(Strary,dtype="float32")

def LoadOneImage(imgfile):
    img = cv2.imread(imgfile,cv2.IMREAD_GRAYSCALE)
    return img        

I,G = Load('D:\\Dataset\\PaperTrain.txt')
Ti,Tg = Load('D:\\Dataset\\PaperTest.txt',False)
with open("ImageData.pkl","wb") as File:
    pickle.dump(I,File)
    pickle.dump(G,File)
    pickle.dump(Ti,File)
    pickle.dump(Tg,File)