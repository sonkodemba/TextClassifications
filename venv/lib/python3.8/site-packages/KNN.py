from numpy import *  
import operator  
from cProfile import label  
import matplotlib  
import matplotlib.pyplot as plt  
from os import listdir  
#create a dataset and labellist  
def createDataSet():  
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])  
    labels = ['A','A','B','B']  
    return group, labels  
#the implimention of KNN  
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()   
    print(distances)
    print(sortedDistIndicies)  #return the index of the sorted distances which can reflect to labels 
    classCount={}          
    for i in range(k):
	print(sortedDistIndicies[i])
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0] 
#change file datingTestSet2 to dataset and labellist      
def file2matrix(filename):  
    fr = open(filename)  
    arrayOLines = fr.readlines()  
    numberOfLines = len(arrayOLines)  
    returnMat = zeros((numberOfLines,3))  
    classLabelVector = []  
    index = 0  
    for line in arrayOLines:  
        line = line.strip()  
        listFromLine = line.split('\t')  
        returnMat[index,:]=listFromLine[0:3]  
        classLabelVector.append(int(listFromLine[-1]))  
        index += 1  
    return returnMat,classLabelVector  
#normlize the dataset
def autoNorm(dataSet):  
    minVals = dataSet.min(0)  #The 0 in dataSet.min(0) allows you to take the minimums from the columns, not the rows
    maxVals = dataSet.max(0)  
    ranges = maxVals - minVals  
    normDataSet = zeros(shape(dataSet))  
    m = dataSet.shape[0]  
    normDataSet = dataSet - tile(minVals,(m,1))  
    normDataSet = normDataSet/tile(ranges,(m,1))  
    return normDataSet, ranges, minVals 
#test KNN with data in datingTestSet2.txt
def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    #print m
    #print numTestVecs
    #print(datingLabels[numTestVecs:m])
    print errorCount
#use KNN to justify if a person the girl like
def classifyPerson():
    resultList =['not at all','in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "You will probably like this person: ",resultList[classifierResult - 1]
#change a imgfile to a two-dimination array which have one row with 1024 numbers
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])#two-dimination array
    return returnVect
#use testDigits to test the knn and be true only when the first number is the same,ie,0_x.txt in test meets with 0_y.txt in training
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))

'''
group,labels=createDataSet()
classify0([0.5,0.5], group, labels, 3)
datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')  
normDataSet, ranges, minVals=autoNorm(datingDataMat)
print(normDataSet[0:50])
datingClassTest()
classifyPerson()
filename='trainingDigits/0_1.txt'
returnVect=img2vector(filename)
print(len(returnVect[0]))
handwritingClassTest()
fig = plt.figure()  
ax = fig.add_subplot(1,1,1)  
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))  
plt.show()  
'''
