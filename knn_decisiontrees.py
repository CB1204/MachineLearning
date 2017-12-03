# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 16:47:10 2017

@author: Christian
"""


#decision tree algorithms
import numpy as np
import csv

def decisiontree():
    filename = "C:/Users/Christian/Documents/Master/2017_WS/IN2064-MachineLearning/Hw1/01_homework_dataset.csv"
    data = readdata(filename)[1:] #remove first element (label names)
    for idx, line in enumerate(data):
        line = [float(num) for num in line]
        data[idx]=line
    data=np.array(data)
    
    tree=dtmaker(data,[], 0,0,2, '-')
    return tree

def dtmaker(data, tree, ref, depth, maxDepth, decision):
    #creates a decision tree based on input data 
    # number of input dimensions = columns of data -1
    # number of test samples = rows of data
    # output label size is assumed to be 0
    
    # save values for best dimension, value, and corresponding gini improvement
    bestdim = 0
    bestval = 0
    bestgini = 1
    
    # compute current gini value
    gini = singleGini(data)
    if gini==0 or depth>=maxDepth:
        tree.append([data,-1,0,gini,ref,decision])
        return tree
    
    # loop through all dimensions
    for dim in range(np.size(data,1)-1):
        for val in data[:,dim]:
            # split at each possible value, find gini, and if better than best, overwrite
            newGini=giniIndex(data,dim,val)

            if newGini < bestgini:
                #save actvalue as best value
                bestdim=dim
                bestval=val
                bestgini=newGini

    [data1, data2] = performsplit(data,bestdim, bestval)  
    
    #           data which was split, dimension, along which the split happened
    #           value for split, gini index at split, parent reference
    tree.append([data, bestdim, bestval, singleGini(data), ref,decision])
        
    #continue developing
    newref=len(tree)-1
    tree=dtmaker(data1, tree, newref, depth+1, maxDepth, 'yes')
    tree=dtmaker(data2, tree, newref, depth+1, maxDepth, 'no')
    
    return tree

def giniIndex(data, splitdim, splitval):
    # computes the gini score of a split (only of the two splitted nodes)
    numPts = np.size(data,0)
    [data1, data2]=performsplit(data, splitdim, splitval)
    numPts1 = np.size(data1,0)
    numPts2 = np.size(data2,0)
    
    nodegini = numPts1/numPts*singleGini(data1)+numPts2/numPts*singleGini(data2)
    return nodegini

def singleGini(data):
    # computes the gini score for the input dataset (not considering any splits)
    
    #programmed according to slide 23, chapter 3
    labels=data[:,-1]
    sumPts=np.size(data,0)
    if sumPts==0:
        return 0.5
    LabelCount = mostcommon(labels, returnAll=True)
    gini = 1
    for label, numPts in LabelCount:
        gini = gini-(numPts/sumPts)**2
    
    return gini
    
def performsplit(data, splitdim,splitval):
    # computes the split of the data set and returns data1 (<=splitval), data2 (>splitval)
    # splitdim is the integer number of the split dimension
    # splitval denotes the value at which the split is carried out
    # dataIndex is the value for the 
    
    #prepare a binary vector to be able to easily split data
    selectdat1=np.less_equal(data[:,splitdim],splitval)
    data1=data[selectdat1,:]
    data2=data[np.invert(selectdat1),:]
    
    return [data1, data2]

def countitems(data):
    sumdct=dict()
    for el in data:
        sumdct[str(el)]=sumdct.get(str(el),0)+1
    return sumdct

def readdata(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        lstOut = list(reader)
            
    return lstOut

def mostcommon(inlst, returnAll=False):
    sumdct=dict()
    print(inlst)
    maxval=0
    maxclass=0.
    for el in inlst:
        newval=sumdct.get(str(el),0)+1
        sumdct[str(el)]=newval
        if newval>maxval:
            maxval=newval
            maxclass=el
    if returnAll == False:
        #return maximum class and value
        return [maxclass, maxval]
    else:
        # return all classes and count
        outLst=[]
        for key in sumdct.keys():
            outLst.append([float(key),sumdct[key]])
        return outLst

#-----------------------------------------------------------------------------  
#-----------------------------------------------------------------------------------------
#kNN algorithm

def knn(data, k, sample, classify=True):
    # performs knn - can output classification or regression result, based on parameter classify
    
    #first: distance computation
    #assumption: sample is a row-vector
    numPts = np.size(data,0)
    dist=[0]*numPts
    for i in range(numPts):
        for j in range(np.size(sample)):
            dist[i]=dist[i]+(data[i,j]-sample[j])**2
        dist[i]=np.sqrt(dist[i])

    #now, find elements with the lowest distance
    ind = [0]*k
    finalDist=[0]*k
    for i in range(k):
        ind[i]=dist.index(min(dist))
        finalDist[i]=dist[ind[i]]
        dist[ind[i]]=max(dist)+1
            
    clabels = data[ind,-1]
    print(dist)
         
    if classify:
        #evaluate and return most common element
        return mostcommon(clabels)[0]
    else:
        #evaluate, calculate weighted average
        out=0
        sumdist=sum([1/el for el in finalDist])
        for i in range(k):
            out+=clabels[i]/finalDist[i]
        out=out/sumdist
            
        return out
        
  
def test():

    filename = "C:/Users/Christian/Documents/Master/2017_WS/IN2064-MachineLearning/Hw1/01_homework_dataset.csv"
    data = readdata(filename)[1:] #remove first element (label names)
    for idx, line in enumerate(data):
        line = [float(num) for num in line]
        data[idx]=line
    data=np.array(data)
    print(data)
    print(knn(data,3,np.array([4.1,-0.1,2.2])))
    print(knn(data,3,np.array([6.1,0.4,1.3])))
    print(knn(data,3,np.array([4.1,-0.1,2.2]), classify=False))
    print(knn(data,3,np.array([6.1,0.4,1.3]), classify=False))

    tree=decisiontree()
        
    print(tree)
test()

    