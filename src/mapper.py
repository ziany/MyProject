import preprocessing
import ngramGenerator
import polarity
import features

stopWords = preprocessing.getStopWordList('../resources/stopWords.txt')
slangs = preprocessing.loadSlangs('../resources/internetSlangs.txt')
sentiWordnet=polarity.loadSentiFull('../resources/sentiWordnetBig.csv')
emoticonDict=features.createEmoticonDictionary("../resources/emoticon.txt")

# do the preprocessing here and 3 output files
# done in the threeFileGen script 

# define here lists of unigram for each file , 3 lists
pos=ngramGenerator.mostFreqList('../data/positive_processed.csv',2)
positive=[w[0] for w in pos]
neg=ngramGenerator.mostFreqList('../data/negative_processed.csv',2)
negative=[w[0] for w in neg]
neu=ngramGenerator.mostFreqList('../data/neutral_processed.csv',2)
neutral=[w[0] for w in neu]

total=positive+negative+neutral # total unigram vector
#print len(total)


# prepare mapping function

def mapper(filename,label):
#    k=0
    f=open(filename,'r')
    line=f.readline()
    
    while line:
#        k=k+1
        newLine=preprocessing.processTweet(line,stopWords,slangs)
        line=newLine
        out=label+'\t'
        
        p=polarity.polarity(line,sentiWordnet)
        out=out+str(p[0])+'\t'+str(p[1])+'\t'+str(p[2])+'\t' # aggregate polarity for pos neg and neutral 
#        print len(out.split('\t'))
        pos=polarity.posFreq(line,sentiWordnet)
        out=out+str(pos['v'])+'\t'+str(pos['n'])+'\t'+str(pos['a'])+'\t'+str(pos['r'])+'\t' # pos counts inside the tweet
        out=out+str(features.emoticonScore(line,emoticonDict))+'\t' # emo aggregate score be careful to modify weights
        out=out+str(len(line))+'\t' # for the length
        out=out+str(features.upperCase(line))+'\t' # uppercase existence : 0 or 1
        
        out=out+str(features.exclamationTest(line))+'\t'
#        print len(out.split('\t'))
        out=out+str(line.count("!"))+'\t'
       
        out=out+str(features.questionTest(line))+'\t'
        
        out=out+str(line.count('?'))+'\t'
        
        out=out+str(features.freqCapital(line))+'\t'
#        print len(out.split('\t'))
        for w in total:  # unigram
            if (w in line):
                out=out+'1\t'
            else:
                out=out+'0\t'
        
        
        fo.write(out+'\n')
#        print len(out.split('\t'))
#        k=k+1
#        print str(k)+' line(s) mapped'
        line=f.readline()
    f.close()
    return None

# map a tweet to a vector here
print "Mapping negative file..."
fo=open('../data/bigVectorTweets.csv','w')
mapper('../data/negative_processed.csv','0')
print 'negative file mapped'


print "Mapping positive file ..."
mapper('../data/positive_processed.csv','4')
print 'positive file mapped'


print "Mapping neutral file..."
mapper('../data/neutral_processed.csv','2')
print 'neutral file mapped'

fo.close()




# write the list of vectors in the same file

# you are done 
print "mapping done"


#fi=open('../data/test.txt','r')
#line=fi.readline()
#while line:
#    processed=preprocessing.processTweet(line,stopWords,slangs)
#    print polarity.polarity(processed,sentiWordnet)
#    line=fi.readline()

#fi.close()

#print ngramGenerator.mostFreqList('../data/test.txt',5)
#  ngram for each class 



