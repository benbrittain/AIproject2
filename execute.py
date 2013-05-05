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
        print(test[2], nn.classify(test))

def getTestSet(fileString):
    fin = open(fileString)
    testVectors = []
    for line in fin:
        aList = line.strip().split(',')
        testVectors.append((float(aList[0]),float(aList[1]),int(aList[2])))
    return testVectors

    
main()
