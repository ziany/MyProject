# python script for determining the polarity and POS characteristics
# of an input tweet using SentiWordNet3.0 dictionnary
from __future__ import division
import features
# load input file in a dictionnary
def loadSentiSimple(filename):
    output={}
    print "Opening SentiWordnet file..."
    fi=open(filename,"r")
    line=fi.readline() # skip the first header line
    line=fi.readline()
    print "Loading..."

    while line:
        l=line.split('\t')
        tag=l[0]
        word=l[1]
        pos=abs(float(l[3]))
        neg=abs(float(l[4]))
        neu=abs(float(l[5]))

        output[word]=[tag,pos,neg,neu]
        line=fi.readline()
    fi.close()
    return output

def loadSentiFull(filename): # need fixing , use loadSentiSmall instead 
    output={}
    print "Opening SentiWordnet file..."
    fi=open(filename,"r")
    line=fi.readline() # skip the first header line
    line=fi.readline()
    print "Loading..."

    while line:
        l=line.split('\t')
        try:
            tag=l[0]
            sentence=l[4]
            new = [word for word in sentence.split() if (word[-2] == "#" and word[-1].isdigit())]
            pos=abs(float(l[2]))
            neg=abs(float(l[3]))
            neu=float(1-pos-neg)
        except:
#            print line
            line=fi.readline()
            continue

        for w in new:
            output[w[:-2]]=[tag,pos,neg,neu]
        line=fi.readline()
    fi.close()
    return output

def polarity(tweet,sentDict): # polarity aggregate of a tweet from sentiWordnet dict
    pos=0.0
    neg=0.0
    neu=0.0
    n_words=0
    for w in tweet.split():
        if w in sentDict.keys():
            n_words=n_words+1
            pos=pos+sentDict[w][1]
            neg=neg+sentDict[w][2]
            neu=neu+sentDict[w][3]
        if features.hashTest(w) and w[1:] in sentDict.keys():
            pos=pos+2*sentDict[w[1:]][1] # more weight for hashed words
            neg=neg+2*sentDict[w[1:]][2]
            neu=neu+2*sentDict[w[1:]][3]
            
    if (n_words ==0 ):
        return [pos,neg,neu]
    else:
        return [pos/n_words,neg/n_words,neu/n_words]


def posFreq(tweet,dict): # calculates the frequency of apperances of pos in a tweet
    result={}
    result['v']=0
    result['n']=0
    result['a']=0
    result['r']=0
    nbr=0
    for w in tweet.split():
        if (w in dict.keys()):
            nbr=nbr+1
            result[dict[w][0]]=result[dict[w][0]]+1
    if (nbr != 0):
        result['v']=result['v']/nbr
        result['a']=result['a']/nbr
        result['n']=result['n']/nbr
        result['r']=result['r']/nbr
    return result

