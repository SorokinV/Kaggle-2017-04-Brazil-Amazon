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

# Построение решения 0-1 исходя из пределов прохождения cc (построение через getTh)
def getProbX01 (tPr, cc, weather=[0,3,9,10], cloudy=9) :
    t01  = np.zeros(tPr.shape,dtype=np.uint8);
    temp0 = tPr[:,weather].argmax(axis=1); #print(temp0[0:10])
    temp1 = np.array(weather)[temp0]
    for i in range(t01.shape[0]) : 
        temp2 = tPr[i,:]>cc
        t01[i,temp2]    = 1
        t01[i,weather]  = 0
        t01[i,temp1[i]] = 1
        if t01[i,cloudy]==1 : t01[i,:]=0; t01[i,cloudy]=1;
    
    #print(temp1[0:10])
    #print(t01[0:10,weather+[1,2]])
    return(t01)

# Построение пределов прохождения для предсказанного
def getTh (tGround,tPredict, prec=100 ) :
    ixx,iacc = [], []
    for i in range(tGround.shape[1]) :
        max, maxxx = 0.0, 0.0
        for xx in range(0,prec+1) :
            tempYP = getProb01(tPredict[:,i],th=(float(xx)/prec)); #print(tPredict[:,1],tempYP[0])
            temp = skm.accuracy_score(tGround[:,i],tempYP)
            #print(float(xx/10.0),temp)
            if (temp>max) : 
                max = temp; maxxx = float(xx)
                temp = skm.confusion_matrix(tGround[:,i],tempYP);
                minloss = temp[0][1]+temp[1][0]
        ixx.append(maxxx)
        tempYP = getProb01 (tPredict[:,i])
        temp = skm.accuracy_score(tGround[:,i],tempYP)
        minloss05 = skm.confusion_matrix(tGround[:,i],tempYP);
        minloss05 = minloss05[0][1]+minloss05[1][0]
        iacc.append((maxxx,max,temp,minloss,minloss05))
    ixx = [float(xx/prec) for xx in ixx]
    return (ixx,iacc)
##ixx, iacc = getTh(trY,trP, prec=100)
#np.array(ixx) , iacc
    
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