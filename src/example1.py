# the actual classifier script for predicting a sentiment using SVM
from __future__ import division
from sklearn import svm
from sklearn import cross_validation

import numpy as np


import features
import polarity
import ngramGenerator
import preprocessing

print "Initializing dictionnaries"
stopWords = preprocessing.getStopWordList('../resources/stopWords.txt')
slangs = preprocessing.loadSlangs('../resources/internetSlangs.txt')
sentiWordnet=polarity.loadSentiFull('../resources/sentiWordnetBig.csv')
emoticonDict=features.createEmoticonDictionary("../resources/emoticon.txt")

print "Bulding 150 unigram vector"
positive=ngramGenerator.mostFreqList('../data/positive_processed.csv',50)
negative=ngramGenerator.mostFreqList('../data/negative_processed.csv',50)
neutral=ngramGenerator.mostFreqList('../data/neutral_processed.csv',50)


total=positive+negative+neutral # total unigram vector
#print total
total=[]
def mapTweet(tweet,sentiWordnet,emoDict,unigram,slangs):
    out=[]
    line=preprocessing.processTweet(tweet,stopWords,slangs)
   
    p=polarity.polarity(line,sentiWordnet)
   
    out.extend([float(p[0]),float(p[1]),float(p[2])]) # aggregate polarity for pos neg and neutral here neutral is stripped
    pos=polarity.posFreq(line,sentiWordnet)
    out.extend([float(pos['v']),float(pos['n']),float(pos['a']),float(pos['r'])]) # pos counts inside the tweet
#    out.append(float(features.emoticonScore(line,emoDict))) # emo aggregate score be careful to modify weights
    out.append(float(len(features.hashtagWords(line)))) # number of hashtagged words
    out.append(float(len(line)/140)) # for the length normalized
    out.append(float(features.upperCase(line))) # uppercase existence : 0 or 1
    out.append(float(features.exclamationTest(line)))
    out.append(float(line.count("!")))
    out.append(float((features.questionTest(line))))
    out.append(float(line.count('?')/140)) # normalized
    out.append(float(features.freqCapital(line)))
    for w in unigram:  # unigram
            if (w in line):
                out.append(float(1))
            else:
                out.append(float(0))
    return out
# load matrix
def loadMatrix(posfilename,neufilename,negfilename,poslabel,neulabel,neglabel):
    vectors=[]
    labels=[]
    f=open(posfilename,'r')
    kpos=0
    kneg=0
    kneu=0
    line=f.readline()
    while line:
        kpos=kpos+1
        z=mapTweet(line,sentiWordnet,emoticonDict,total,slangs)
        vectors.append(z)
        labels.append(float(poslabel))
        line=f.readline()
        print str(kpos)+"positive line loaded"+str(len(vectors))+" "+str(len(labels))
    f.close()
    
    f=open(neufilename,'r')
    line=f.readline()
    while line:
        kneu=kneu+1
        z=mapTweet(line,sentiWordnet,emoticonDict,total,slangs)
        vectors.append(z)
        labels.append(float(neulabel))
        line=f.readline()
        print str(kneu)+"neutral lines loaded"
    f.close()
    
    f=open(negfilename,'r')
    line=f.readline()
    while line:
        kneg=kneg+1
        z=mapTweet(line,sentiWordnet,emoticonDict,total,slangs)
        vectors.append(z)
        labels.append(float(neglabel))
        line=f.readline()
        print str(kneg)+"negative lines loaded"
    f.close()
    return vectors,labels

#print "Loading training data"
#X,Y=loadMatrix('../data/positive_processed.csv','../data/neutral_processed.csv','../data/negative_processed.csv','4','2','0')

 # training set and labels
#print len(X[0])
#print len(Y)
# train model 

# do a cross fold validation 
# precision and recall here 

# map tweet into a vector 
def trainModel(X,Y,knel):
    clf=svm.SVC(kernel=knel) # linear, poly, rbf, sigmoid, precomputed , see doc
    clf.fit(X,Y)
    return clf

def predict(tweet,model): # test a tweet against a built model 
    z=mapTweet(tweet,sentiWordnet,emoticonDict,total,slangs) # mapping
    return model.predict([z]).tolist() # transform nympy array to list 

def loadTest(filename): # function to load test file in the csv format : sentiment,tweet 
    f=open(filename,'r')
    line=f.readline()
    labels=[]
    vectors=[]
    while line:
        l=line[:-1].split(r'","')
        s=float(l[0][1:])
        tweet=l[5][:-1]

        z=mapTweet(tweet,sentiWordnet,emoticonDict,total,slangs)
        vectors.append(z)
        labels.append(s)
        line=f.readline()
#        print str(kneg)+"negative lines loaded"
    f.close()
    return vectors,labels

def batchPredict(vectors,model): # the output is a numpy array of labels
    return model.predict(vectors).tolist()

def testModel(vectors,labels,model): # for a given set of labelled vectors calculate model labels and give accuract
    a=0 # wrong classified vectors
    newLabels=model.predict(vectors).tolist()
    for i in range(0,len(newLabels)):
        if newLabels[i]!=labels[i]:
            a=a+1
    if len(labels)==0:
        return 0.0
    else:
        return 1-a/len(labels) # from future import dividion


# loading training data
print "Loading training data"
X,Y=loadMatrix('../data/positive_processed.csv','../data/neutral_processed.csv','../data/negative_processed.csv','4','2','0')

# Optimizing weights
def weight(x,y): # weights an input vector function 
     r=[]
     def simple(z,t):
         rr=[]
         for i in range(0,len(z)):
             rr.append(z[i]*t[i])
         return rr
     for f in x:
         r.append(simple(f,y))
     return r

w=[0.2*i for i in range(0,6)] # possible weights for a single feature

# 5 fold cross validation
x=np.array(X)
y=np.array(Y)
clf = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score(clf, x, y, cv=5)
print scores # the precision for five iterations
print("Accuracy of the model using 5 fold cross validation : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))# Actual testing 
MODEL=trainModel(X,Y,'linear') # poly of degree 3 (default)

 


V,L=loadTest('../data/test_dataset.csv')

#Z=batchPredict(V,MODEL) # output of batch prediction
# here you can redirect to a csv file 

print "Classification done : Performance over test dataset : "+str(testModel(V,L,MODEL))

print "done"
