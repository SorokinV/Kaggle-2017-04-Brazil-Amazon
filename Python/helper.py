import sys,os,datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score
import  cv2 as cv

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
    
    return(trLabels,trDirTIF,trDirJPG,teDirTIF,teDirTIF)

def paths_save_load () :
    return()

def formFH (nf, bins, printOK=False) :
    nx = None
    try : 
        ni = cv.imread(nf,-1); 
        if (ni is not None) :
            if not ((ni.shape[2]==3) or (ni.shape[2]==4)) and printOK : print('----- error ---- shape:',ni.shape,nf)
            if (ni.shape[2]==3) :
                r,g,b = ni[:,:,0],ni[:,:,1],ni[:,:,2]
                rh = np.divide(np.histogram(r,bins=bins,density=False)[0],(0.0+r.size))
                gh = np.divide(np.histogram(g,bins=bins,density=False)[0],(0.0+g.size))
                bh = np.divide(np.histogram(b,bins=bins,density=False)[0],(0.0+b.size))
                nx = np.hstack((rh,gh,bh)); 
            if (ni.shape[2]==4) :
                r,g,b,n = ni[:,:,2],ni[:,:,1],ni[:,:,1],ni[:,:,3]
                dv = np.divide((r-n),(r+n+0.01))
                dw = np.divide((g-n),(g+n+0.01))
                rh = np.divide(np.histogram(r,bins=bins,density=False)[0],(0.0+r.size))
                gh = np.divide(np.histogram(g,bins=bins,density=False)[0],(0.0+g.size))
                bh = np.divide(np.histogram(b,bins=bins,density=False)[0],(0.0+b.size))
                nh = np.divide(np.histogram(n,bins=bins,density=False)[0],(0.0+n.size))
                dvh= np.divide(np.histogram(dv,bins=bins,density=False)[0],(0.0+dv.size))
                dwh= np.divide(np.histogram(dw,bins=bins,density=False)[0],(0.0+dw.size))
                nx = np.hstack((rh,gh,bh,nh,dvh,dwh)); 
    except BaseException as e :
        print(nf,e); nx = None;
    
    if (nx is None) and printOK : print('------ None:',nf)
        
    return(nx)


def formFHMMM (nf, bins, mmm, printOK=False) :
    nx = []
    try : 
        ni = cv.imread(nf,-1); 
        if (ni is not None) :
            if not ((ni.shape[2]==3) or (ni.shape[2]==4)) and printOK : print('----- error ---- shape:',ni.shape,nf)
            if (ni.shape[2]==3) :
                r,g,b = ni[:,:,0],ni[:,:,1],ni[:,:,2]
                for im,mm in zip([r,g,b],mmm.tolist()) :
                    im1 = np.clip(im.astype(np.float32),mm[0],mm[1])
                    im1 = (im1-mm[0])/(mm[1]-mm[0])*bins;
                    hh1 = np.divide(np.histogram(im1,bins=bins,density=False)[0],(0.0+im1.size))
                    nx.append(hh1)
                nx = np.array(nx).flatten(); 
            if (ni.shape[2]==4) :
                r,g,b,n = ni[:,:,2],ni[:,:,1],ni[:,:,1],ni[:,:,3]
                dv = np.divide((r-n),(r+n+0.01))
                dw = np.divide((g-n),(g+n+0.01))
                for im,mm in zip([r,g,b,n,dv,dw],mmm.tolist()) :
                    im1 = np.clip(im.astype(np.float32),mm[0],mm[1])
                    im1 = (im1-mm[0])/(mm[1]-mm[0])*bins;
                    hh1 = np.divide(np.histogram(im1,bins=bins,density=False)[0],(0.0+im1.size))
                    nx.append(hh1)
                nx = np.array(nx).flatten(); 
    except BaseException as e :
        print(nf,e); nx = None;
    
    if len(nx)==0 and printOK : 
        print('------ None:',nf); nx = None
        
    return(nx)