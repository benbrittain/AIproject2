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
    for test in testData:
        print test[0],test[1]
        print(test[2], nn.classify(test[0],test[1]))

    generateClassification(nn)


def generateClassification(nn):
    bolt = [[],[]]
    nut = [[],[]]
    ring = [[],[]]
    junk= [[],[]]
    
    fine = 1000 
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
