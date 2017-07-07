# -*- coding: utf-8 -*-

import numpy as np
import sklearn.metrics as skm

# Процедуры для оценки результативности

def confusion_matrix (tTruth, tPredict) :
    tp  = (tTruth==tPredict)&(tTruth==1); tp=len(tp[tp])
    fn  = (tTruth==tPredict)&(tTruth==0); fn=len(fn[fn])
    tn  = (tTruth<>tPredict)&(tTruth==1); tn=len(tn[tn])
    fp  = (tTruth<>tPredict)&(tTruth==0); fp=len(fp[fp])
    return(np.array([[tp,tn],[fp,fn]]))

def getConfusion (tGroundTrue, tPredict) :
    cList = []
    for i in range(tPredict.shape[1]) :
        #cList.append(skm.confusion_matrix(tGroundTrue[:,i],tPredict[:,i]))
        cList.append(confusion_matrix(tGroundTrue[:,i],tPredict[:,i]))
    return(np.array(cList))

def getRocAUC (tGroundTrue, tPredict) :
    rList = []
    for i in range(tPredict.shape[1]) :
        rList.append(skm.roc_auc_score(tGroundTrue[:,i],tPredict[:,i]))
    return(np.array(rList))
    
def getProb01 (trYP, th=0.5) :
    trYY = trYP.copy()
    trYY[trYY<th] = 0
    trYY[trYY>0]  = 1
    return (trYY)
    
#
# Оценка результативности предсказания
#
# Выдача массива:
#   номер поля
#   accuracy
#   roc_auc
#   tt, nn        <-- должно быть
#   tp,fn,(fp+tn) <-- получили
#
#
def estimateResult (tThruth, tPredict, printOK=False) :
    res = [];
    for i in range(tThruth.shape[1]) :
        cm = confusion_matrix(tThruth[:,i],tPredict[:,i])
        if printOK :
            print('{} acc={} roc={} not={} yes={} no={} true={} all-1-0=({:.4f}-{:.4f})'.format(i,
                      skm.accuracy_score(tThruth[:,i],tPredict[:,i]),
                      skm.roc_auc_score(tThruth[:,i],tPredict[:,i]),
                      cm[0,1]+cm[1,0],
                      cm[0,0],cm[1,1],
                      cm[0,0]+cm[1,1],
                      float(cm[0,0])/len(tThruth[tThruth[:,i]==1,i]),
                      float(cm[1,1])/len(tThruth[tThruth[:,i]==0,i]),
                     ));
        res.append ((i,tThruth.shape[0],
                    skm.accuracy_score(tThruth[:,i],tPredict[:,i]),
                    #skm.accuracy_score(tThruth[:,i],tPredict[:,i]),
                    skm.roc_auc_score(tThruth[:,i],tPredict[:,i]),
                    len(tThruth[tThruth[:,i]==1]), len(tThruth[tThruth[:,i]==0]),
                    cm[0,0], cm[1,1],cm[0,1]+cm[1,0]))
    return(np.array(res))