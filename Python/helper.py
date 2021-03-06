# -*- coding: utf-8 -*-

import sys,os,datetime
import numpy as np
import pandas as pd
import cv2 as cv

dirInput = '../Data'

def paths_input () :
    trDirTIF = os.path.join(dirInput,'train-tif-v2');
    trDirJPG = os.path.join(dirInput,'train-jpg');
    trLabels = os.path.join(dirInput,'train_v2.csv');
    teDirTIF = os.path.join(dirInput,'test-tif-v2');
    teDirJPG = os.path.join(dirInput,'test-jpg-v2');
    
    assert os.path.exists(dirInput), "The {} folder does not exist".format(dirInput)
    assert os.path.exists(trLabels), "The {} folder does not exist".format(trLabels)
    assert os.path.exists(trDirTIF), "The {} folder does not exist".format(trDirTIF)
    assert os.path.exists(trDirJPG), "The {} folder does not exist".format(trDirJPG)
    assert os.path.exists(teDirTIF), "The {} folder does not exist".format(teDirTIF)
    assert os.path.exists(teDirJPG), "The {} folder does not exist".format(teDirJPG)
    
    return(trLabels,trDirTIF,trDirJPG,teDirTIF,teDirJPG)

def paths_save_load () :
    return()

#
# Построение по входным наборам гистограмм и переработанных изображений (изменение размеров, улучшение, сглаживание
#

def formX3 (ni, resize=(32,32), printOK=False, GaussianOK=True, EqualizeOK=False) :
        if GaussianOK : ni = np.array([cv.GaussianBlur(ni[:,:,i],(3,3),0) for i in range(ni.shape[2])]).T;
        if EqualizeOK : ni = np.array([cv.equalizeHist(ni[:,:,i]) for i in range(ni.shape[2])]).T;
        if resize and ((ni.shape[0],ni.shape[1])<>resize) : ni = cv.resize(ni,resize)
        return(ni)
    
def formX4 (ni, resize=(32,32), printOK=False, GaussianOK=True, EqualizeOK=False, OnlyNI=False, ExtIndex=True) :
    
        #before or after???? 
        #Equalize only 256 color!  ni = np.array([cv.equalizeHist(ni[:,:,i]) for i in range(ni.shape[2]-1)]).T;
        
        if GaussianOK : ni = np.array([cv.GaussianBlur(ni[:,:,i],(3,3),0) for i in range(ni.shape[2])]).T;
        r,g,b,n = ni[:,:,2],ni[:,:,1],ni[:,:,0],ni[:,:,3]
        if resize and ((ni.shape[0],ni.shape[1])<>resize) : 
            r,g,b,n = cv.resize(r,resize),cv.resize(g,resize),cv.resize(b,resize),cv.resize(n,resize)
        r,g,b,n = np.array(r,np.float32),np.array(g,np.float32),np.array(b,np.float32),np.array(n,np.float32)
        
        
        # навал индексной массы
        ndvi,ndwi  = np.divide((r-n),(r+n+0.0001)), np.divide((g-n),(g+n+0.0001))
        ## ? unknown range evi        = 2.5*np.divide((n-r),(n+6*r-7.5*b+1.0))
        evi2       = 2.5*np.divide((n-r),(n+2.4*r+1.0))
        savi       = 1.5*np.divide((n-r),(n+r+1.0))
        
        # аккуратно переводим r,g,b,n из 0..65536 в 0..256 = uint8
        r,g,b,n = np.array(r/256.0,np.uint8), np.array(g/256.0,np.uint8), np.array(b/256.0,np.uint8), np.array(n/256.0,np.uint8)
        if (not OnlyNI) and EqualizeOK : r,g,b   = cv.equalizeHist(r), cv.equalizeHist(g),  cv.equalizeHist(b)

        # аккуратно переводим индекснуюю массу из [-1..1] в 0..256 = uint8
        ndvi,ndwi   = np.array((ndvi+1.0)/2.0*256.0,np.uint8), np.array((ndwi+1.0)/2.0*256.0,np.uint8)
        if ExtIndex :
            ## unknown range ? evi     = np.array((evi+1.0)/2.0*256.0,np.uint8)
            evi2    = np.array((evi2+1.041660)/3.541622*256.0,np.uint8)
            savi    = np.array((savi+1.499977)/3.0*256.0,np.uint8)
            ni      = np.array([r,g,b,n,ndvi,ndwi,evi2,savi]).T if not OnlyNI else np.array([n,ndvi,ndwi,evi2,savi]).T;
        else : ni   = np.array([r,g,b,n,ndvi,ndwi]).T if not OnlyNI else np.array([n,ndvi,ndwi]).T;

        #print('----',r[0,0],g[0,0],b[0,0],n[0,0],dv[0,0],dw[0,0],nx[0,0,5])
        del r,g,b,n,ndvi,ndwi,evi2,savi
        return (ni)

def formImExt (nf, resize=(32,32), printOK=False, OnlyNI=False, GaussianOK=True, EqualizeOK=False, ExtIndex=True) :
    nx = None
    try : 
        ni = cv.imread(nf,-1); 
        if (ni is not None) :
            
            if not ((ni.shape[2]==3) or (ni.shape[2]==4)) and printOK : print('----- error ---- shape:',ni.shape,nf)
                
            if (ni.shape[2]==3) :   nx = formX3 (ni, resize=resize, GaussianOK=GaussianOK, EqualizeOK=EqualizeOK)
                
            if (ni.shape[2]==4) :   nx = formX4 (ni, resize=resize, GaussianOK=GaussianOK, EqualizeOK=EqualizeOK, OnlyNI=OnlyNI, ExtIndex=ExtIndex)

    except BaseException as e : 
        print(nf,e); nx = None;
    
    if nx is None and printOK : 
        print('------ None:',nf); nx = None
        
    return(nx)

def formImHist (nf, count, printOK=False, OnlyNI=False, GaussianOK=True, EqualizeOK=False, ExtIndex=True) :
    
    def histN (nf,bins) :
        h = []
        for i in range(nf.shape[2]) : 
            hh,_ = np.histogram(nf[:,:,i].ravel(),bins=bins)
            h = h + hh.tolist()
        return (np.array(h,dtype=np.uint16))
             
    def calculateBins (low,high,count) :
        size    = float(high-low)/float(count)
        bins = [low+x*size for x in range(count+1)]; #print(count,size,len(bins),bins[:3],bins[-3:])
        return (bins)
             
    nx = None
    try : 
        ni = formImExt (nf, resize=False, printOK=printOK, OnlyNI=OnlyNI, GaussianOK=GaussianOK, EqualizeOK=EqualizeOK, ExtIndex=ExtIndex)
        if (ni is not None) :
            
            if printOK : print('formExtHist 1.1:',nf,ni.shape, ni.min(), ni.max())
            bins = calculateBins(0,255,count)
            if printOK : print('formExtHist 1.2:',count,bins[:4],bins[-4:])
            nx = histN(ni,bins)
            if printOK : print('formExtHist 1.3:',nf,nx.shape)
            
    except BaseException as e :
        print(nf,e); nx = None;
    
    if nx is None and printOK : 
        print('------ None:',nf); nx = None
        
    return(nx)