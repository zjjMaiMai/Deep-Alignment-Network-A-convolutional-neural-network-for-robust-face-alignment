import numpy as np
import cv2
from scipy import ndimage
import os.path
'''
Pts
'''

IMGSIZE = 112
LANDMARK = 68
MIRRORS = True
DATASCALE = 10

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

def ReadPts(path):
    landmarks = np.genfromtxt(path, skip_header=3, skip_footer=1)
    landmarks = landmarks - 1

    return landmarks

def bestFitRect(points, meanS, box=None):
    if box is None:
        box = np.array([points[:,0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()])
    boxCenter = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

    boxWidth = box[2] - box[0]
    boxHeight = box[3] - box[1]

    meanShapeWidth = meanS[:, 0].max() - meanS[:, 0].min()
    meanShapeHeight = meanS[:, 1].max() - meanS[:, 1].min()

    scaleWidth = boxWidth / meanShapeWidth
    scaleHeight = boxHeight / meanShapeHeight
    scale = (scaleWidth + scaleHeight) / 2

    S0 = meanS * scale

    S0Center = [(S0[:, 0].min() + S0[:, 0].max()) / 2, (S0[:, 1].min() + S0[:, 1].max()) / 2]    
    S0 += boxCenter - S0Center

    return S0

def RandomSRT(img,landmark,MeanShape,isTrainSet):
    ImgVec = []
    LandVec = []

    if not isTrainSet:
        FitLandmark = bestFitRect(MeanShape,landmark)
        R,T = transform(landmark,FitLandmark)
        R = np.linalg.inv(R)
        T = np.dot(-T, R)
        Img = ndimage.interpolation.affine_transform(img,R,T[[1,0]],output_shape=(IMGSIZE,IMGSIZE))
        return Img,FitLandmark

    R,T = transform(landmark,MeanShape)
    SrcLandmark = np.dot(landmark,R) + T

    for i in range(DATASCALE):
        angle = np.random.normal(0, 15 * np.pi / 180) 
        offset = [np.random.normal(0,0.05 * IMGSIZE) , np.random.normal(0, 0.05 * IMGSIZE)]
        scaling = np.random.normal(1,0.1)

        r = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]]) * scaling

        #TSR
        Landmark = SrcLandmark + offset
        Landmark = (Landmark - Landmark.mean(axis=0)) * scaling + Landmark.mean(axis=0)
        Landmark = np.dot(Landmark - Landmark.mean(axis=0),r) + Landmark.mean(axis=0)

        R,T = transform(landmark,Landmark)
        R2 = np.linalg.inv(R)
        T2 = np.dot(-T, R2)

        Img = ndimage.interpolation.affine_transform(img,R2,T2[[1,0]],output_shape=(IMGSIZE,IMGSIZE))

        ImgVec.append(Img)
        LandVec.append(Landmark)
    return ImgVec,LandVec
    
def GetMeanShape(Shape=None):
    xmax = Shape[:,0].max()
    ymax = Shape[:,1].max()
    xmin = Shape[:,0].min()
    ymin = Shape[:,1].min()

    xmean = Shape[:,0].mean()
    ymean = Shape[:,1].mean()

    halfsize = int((ymean - ymin) * 2.0)

    xstart = int(xmean - halfsize)
    ystart = int(ymean - halfsize)

    ShapeOut = Shape.copy()

    ShapeOut[:,0] = (ShapeOut[:,0] - xstart) / (2 * halfsize) * IMGSIZE
    ShapeOut[:,1] = (ShapeOut[:,1] - ystart) / (2 * halfsize) * IMGSIZE

    return ShapeOut

def GetMirror(Image,Landmark):

    def SwapIdx(Shape,SrcIdx,ToIdx):
        ShapeOut = Shape.copy()
        for Sidx,Tidx in zip(SrcIdx,ToIdx):
            ShapeOut[Tidx] = Shape[Sidx]
        return ShapeOut

    Landmark[:,0] = Image.shape[1] - Landmark[:,0]
    if LANDMARK == 68:
        SrcIdx = [i for i in range(68)]
        ToIdx = [16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,  #1-17
                 26,25,24,23,22,                                #18-22
                 21,20,19,18,17,#23-27
                 27,28,29,30, #28-31
                 35,34,33,32,31,#32-36
                 45,44,43,42,#37-40
                 47,46,#41-42
                 39,38,37,36,#43-46
                 41,40,#47-48
                 54,53,52,51,50,49,48,#49-55
                 59,58,57,56,55,#56-60
                 64,63,62,61,60,#61-65
                 67,66,65]#66-68
        OutLandmark = SwapIdx(Landmark,SrcIdx,ToIdx)
        OutImage = cv2.flip(Image,1)
        return OutImage,OutLandmark

    if LANDMARK == 36:
        SrcIdx = [i for i in range(36)]
        ToIdx = [6,5,4,7,2,1,0,3,
                 14,13,12,15,
                 10,9,8,11,
                 22,21,20,19,18,17,16,
                 27,26,25,24,23,
                 32,31,30,29,28,
                 35,34,33]
        OutLandmark = SwapIdx(Landmark,SrcIdx,ToIdx)
        OutImage = cv2.flip(Image,1)
        return OutImage,OutLandmark

def ReadList(List,Name,isTrainSet=True,MeanShape=None,ImgMean=None,ImgStd=None):
    File = open(List,'r').readlines()
    ImageFileName = []
    LandmarkFileName = []
    for s in File:
        ImageFileName.append(s[:-1])
        LandmarkFileName.append(s[:-4] + 'pts')
        #For jeloTest
        #ptspath = s[:-4] + 'pts'
        #(filepath,tempfilename) = os.path.split(ptspath)
        #(filename,extension) = os.path.splitext(tempfilename)
        #ptspath = filepath + '\\' + str(int(filename) - 1) + extension
        #LandmarkFileName.append(ptspath)

    
    LandmarkVec = []
    ImageVec = []
    Count = 0

    if isTrainSet:
        CountPts = 0
        MeanShape = 0
        for P in LandmarkFileName:
            s = ReadPts(P)
            if s.shape[0] != LANDMARK:
                continue
            MeanShape += GetMeanShape(s)
            CountPts += 1
        MeanShape = MeanShape / CountPts

    for ImagePath,PtsPath in zip(ImageFileName,LandmarkFileName):
        landmark = ReadPts(PtsPath)
        if landmark.shape[0] != LANDMARK:
            continue
        img = cv2.imread(ImagePath,cv2.IMREAD_GRAYSCALE)
        I,L = RandomSRT(img,landmark,MeanShape,isTrainSet)
        LandmarkVec.append(L)
        ImageVec.append(I)

        if isTrainSet and MIRRORS:
            Imirror,Lmirror = GetMirror(img,landmark)
            I,L = RandomSRT(Imirror,Lmirror,MeanShape,isTrainSet)
            LandmarkVec.append(L)
            ImageVec.append(I)

        Count += 1
        print(Count)

    ImageVec = np.array(ImageVec).astype(np.float32).reshape((-1,IMGSIZE,IMGSIZE,1)) / 255.0
    LandmarkVec = np.array(LandmarkVec).astype(np.float32).reshape((-1,LANDMARK * 2))
    MeanShape = np.reshape(MeanShape,(-1)).astype(np.float32)


    if isTrainSet:
        ImgMean = ImageVec.mean(axis=0)
        ImgStd = np.std(ImageVec,axis=0)
        ImageVec = (ImageVec - ImgMean) / ImgStd
        ToSave = {'Image':ImageVec,'Landmark':LandmarkVec,'MeanShape':MeanShape,'ImgMean':ImgMean,'ImgStd':ImgStd}
        np.savez(Name,**ToSave)
        return MeanShape.reshape((-1,2)),ImgMean,ImgStd
    else:
        ImageVec = (ImageVec - ImgMean) / ImgStd
        ToSave = {'Image':ImageVec,'Landmark':LandmarkVec,'MeanShape':MeanShape,'ImgMean':ImgMean,'ImgStd':ImgStd}
        np.savez(Name,**ToSave)
        return MeanShape.reshape((-1,2)),ImgMean,ImgStd

        
ms,im,imgs = ReadList('D:\\Dataset\\MenpoTrain.txt','MenpoTrain',True)
ms,im,imgs = ReadList('D:\\Dataset\\300WTestChallenging.txt','300WTestChallenging',False,ms,im,imgs)