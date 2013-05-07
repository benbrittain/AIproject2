import sys
import pickle
from train import *

def main():
    if (len(sys.argv) != 3):
        print "./execute.py nn.pkl data.csv"
        sys.exit(1)
    nnFile = sys.argv[1]
    nn = pickle.load(open(nnFile,'rb'))

    testData = getTestSet(sys.argv[2])

    stdOut(nn,testData)
    confusionMatrix(nn,testData)
    print "\n Now Generating Classification Regions, please wait a few seconds \n"
    generateClassification(nn)

def stdOut(nn,samples):
    profit = 0
    count = 0
    profitMatrix = [[0.2, -.07, -.07, -.07],
                    [-.07, 0.15, -.07, -.07],
                    [-.07, -.07, .05, -.07],
                    [-.03,-.03,-.03,-.03]]
    for sample in samples:
        classify = nn.classify(sample[0],sample[1]) -1
        correct = sample[2] -1
        if classify != correct:
            count = count + 1
        profit = profit + profitMatrix[classify][correct]
    print("Classification Errors: " + str(count))
    correct =  (((len(samples)-count)*1.0)/len(samples))
    print("Recognition Rate: " + str((correct*100)) + "%")
    print("Profit Obtained: " + str(profit))
        
    
def confusionMatrix(nn,samples):
    matrix = [[0,0,0,0] for x in range(4)]
    for sample in samples:
        classified = nn.classify(sample[0],sample[1]) - 1
        correct = sample[2] - 1
        matrix[correct][classified] = matrix[correct][classified] + 1

    matrixClass = ["bolt", "nut", "ring", "scrap"]
    rowFormat ="{:>10}" * (len(matrixClass) + 1)
    print rowFormat.format("", *matrixClass)
    for classification, row in zip(matrixClass, matrix):
        print rowFormat.format(classification, *row)
    
def generateClassification(nn):
    bolt = [[],[]]
    nut = [[],[]]
    ring = [[],[]]
    junk= [[],[]]
    
    fine = 750
    for x in range(fine):
        for y in range(fine):
            newX = x/float(fine) 
            newY = y/float(fine) 
            itemClass = nn.classify(newX,newY)
            if (itemClass == 1):
                bolt[0].append(newX)
                bolt[1].append(newY)
            if (itemClass == 2):
                nut[0].append(newX)
                nut[1].append(newY)
            if (itemClass == 3):
                ring[0].append(newX)
                ring[1].append(newY)
            if (itemClass == 4):
                junk[0].append(newX)
                junk[1].append(newY)

    plt.plot(bolt[0],bolt[1],'r,')
    plt.plot(nut[0],nut[1],'b,')
    plt.plot(ring[0],ring[1],'g,')
    plt.plot(junk[0],junk[1],'y,')

    plt.title('Classification Regions')
    plt.ylabel('Symmetry')
    plt.xlabel('Eccentricity')

    plt.show()
def getTestSet(fileString):
    fin = open(fileString)
    testVectors = []
    for line in fin:
        aList = line.strip().split(',')
        testVectors.append((float(aList[0]),float(aList[1]),int(aList[2])))
    return testVectors

    
main()
