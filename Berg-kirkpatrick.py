#==========================================
# Title:  Check the behavior  of p-value with Berg-Kirkpatrick pseudo code.
# Author: Marie Dubremetz
# Date:   16 Aug 2017
#==========================================
import random
import numpy as np
from scipy import stats
#Super small data set, certainly not significant:
#A=[1.0,1.0,0.0,1.0]
#B=[0.0,1.0,1.0,0.0]
#>>>"p=0.18"
#less small data set, same perf as before, certainly not significant either:
A=[1.0,1.0,0.0,1.0,1.0,1.0,0.0,1.0]
B=[0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0]
#>>>"P=0.14 #hence the bigger the  original sample the smaller the p value.
#Continue playing around making other samples, vary the size of them and where you put the False/True (0.0/1.0) instances: as the test is paired that has an influence too!
print "Welcome! To this implementation of Berg-Kirkpatrick paper. \n Let's explore the p value according to data size sample."
print "Here are two system output samples\nA=[1.0,1.0,0.0,1.0,1.0,1.0,0.0,1.0] and B=[0.0,1.0,1.0,0.0,0.0,1.0,1.0,0.0]"
print "1.0 means that the system guessed classified correctly the instance. 0.0 means that it classified it incorrectly."
print "that means that , 6 instances were classified correctly by A and  only 4 by B."
n=len(A)
print "Your data set has,",n,"instances."
perfA=float(float(sum(A))/float(n))
perfB=float(float(sum(B))/float(n))
deltaX=perfA-perfB
print "The performance of system A is",perfA
print "The performance of system B is",perfB
print "So delta(X) is",deltaX
deltaX2=2.0*deltaX
print "The DeltaX2 is",deltaX2
numLow=0
numHigh=len(B)-1


listOfDataSetsA=[]
listOfDataSetsB=[]
b=39000
print "Now we generate randomly",b,"samples with resampling"
for y in range(b):
	listOfNumbers=[]#This will be a list of int that will be the indexes of the instances we randomly pick
	for x in range (0, n):
    		listOfNumbers.append(random.randint(numLow, numHigh))
	listOfDataSetsA.append([A[z] for z in listOfNumbers])#We pick the  same instances in data sets A and B so that the test is paired.
	listOfDataSetsB.append([B[z] for z in listOfNumbers])

def score(sample):
	perf=float(float(sum(sample))/float(n))
	return perf
dataSets=[listOfDataSetsA,listOfDataSetsB]
s=0
for i in range(len(dataSets[0])):
	sampleXiA=dataSets[0][i]
	sampleXiB=dataSets[1][i]
	perfA=score(sampleXiA)
	perfB=score(sampleXiB)
	deltaXi=perfA-perfB
	#print deltaXi
	#print "perfA",perfA
	#print "perfB",perfB
	if deltaXi>deltaX2:
		s+=1
print "s=",s
print "b=",b
p=float(s)/float(b)
print "p=",p

print "This was a basic example based on binary evaluation of classifier, but this method works with actually any evaluation metrics.\n****Let's try with Average Precision***"
		

#Your data set is:
gold=[1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0]
A=[0.2,0.0,0.0,0.6,0.0,0.0,0.5,1.0,1.0]
B=[0.2,0.0,0.6,0.6,0.0,0.0,0.7,1.0,1.0]
#>>>"p= 0.342"
gold=gold*10
A=A*10
B=B*10
#>>> p= 0.119230769231
gold=gold*10
A=A*10
B=B*10
#p= 2.5641025641e-05
gold=gold[:300]
A=A[:300]
B=B[:300]
#p= 0.0112051282051
# Your data set has, 300 instances.
# The performance of system A is 0.46379444505
# The performance of system B is 0.444654584695
# So delta(X) is 0.0191398603558
noise=[0.0]*100
gold=gold+noise
A=A+noise
B=B+noise
#p= 0.0103846153846 le noise n'a pas change la perf mais a change le p value, avoir plus d'instances avec les memes resultats minimise la diferencez de perf.


def ap(y,y_scores,verbose=False):
	#TODO: for git
	"""
	array, array-> float
	Given the true positive and their score for class 1.0 (True class), returns the average precision.
	[1,0,1],[0.9,0.8,0.1]
	->0.8333
	"""
	#First convert the list of proba scores y_scores into a list of ranks
	#Communicate it on github?
	
	y_scores= np.asarray([-x for x in y_scores])
	ranks=stats.rankdata(y_scores,method='min')#rank automatically in the "competition" ranking

	
	i_true=0.0
	precs=[]
	i_rank=0
	only_true_ranks=[]
	for instance in y:
		
		if instance==1.:
			i_true+=1.0
			only_true_ranks.append(ranks[i_rank])
		i_rank+=1
	only_true_ranks.sort()
	if verbose==True:
		print "the true pos are:" ,only_true_ranks
	i=0
	for true_rank in only_true_ranks:
			
		
		prec=(i+1)/float(true_rank)
		
		precs.append(prec)
		i+=1
	
	avP=np.mean(precs)
	return avP

n=len(A)
print "Your data set has,",n,"instances."
perfA=ap(gold,A)
print "The performance of system A is",perfA
perfB=ap(gold,B)
deltaX=perfA-perfB

print "The performance of system B is",perfB
print "So delta(X) is",deltaX
deltaX2=2.0*deltaX
print deltaX2
numLow=0
numHigh=len(B)-1


listOfScoresA=[]
listOfScoresB=[]
listOfGolds=[]
b=39000

print "We generate randomly",b,"samples with resampling"
for y in range(b):
	listOfNumbers=[]#This will be a list of int that will be the indexes of the instances we randomly pick
	for x in range (0, n):
    		listOfNumbers.append(random.randint(numLow, numHigh))
	listOfGolds.append([gold[z] for z in listOfNumbers])#We pick the  same instances in data sets A and B so that the test is paired.
	listOfScoresB.append([B[z] for z in listOfNumbers])
	listOfScoresA.append([A[z] for z in listOfNumbers])#We pick the  same instances in data sets A and B so that the test is paired.

dataSets=[listOfScoresA,listOfScoresB,listOfGolds]


#print "Here an example of resampling:",dataSets[2][5],dataSets[0][5],dataSets[1][5]
s=0
for i in range(len(dataSets[0])):
	gold=dataSets[2][i]
	sampleXiA=dataSets[0][i]
	sampleXiB=dataSets[1][i]
	perfA=ap(gold,sampleXiA)

	perfB=ap(gold,sampleXiB)
	deltaXi=perfA-perfB
	
	#print "perfA",perfA
	#print "perfB",perfB

	if deltaXi>deltaX2:
		s+=1
print "s=",s
print "b=",b
p=float(s)/float(b)
print "p=",p

