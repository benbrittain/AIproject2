from pylab import *
import pickle
import random 
import math

class NeuralNet():
    def __str__(self):
        stri = "\n"
        stri = stri + "Input Layer" + str(self.nodesI)
        stri = stri + "\nHidden Layer" + str(self.nodesH)
        stri = stri + "\nOutput Layer" + str(self.nodesO)
        stri = stri + "\n\nFirst weight matrix \n" + '\n'.join(map(str, self.weightsI))
        stri = stri + "\n\nSecond weight matrix \n" + '\n'.join(map(str, self.weightsH))

        stri = stri + "\n\nFirst error matrix \n" + '\n'.join(map(str, self.errorsI))
        stri = stri + "\n\nSecond error matrix \n" + '\n'.join(map(str, self.errorsH))

        return stri
    def __init__(self, inp,hidden,out):
        self.inp = inp + 1
        self.hidden = hidden + 1
        self.out = out

        self.nodesI = [1.0] * (inp + 1)
        self.nodesH = [1.0] * (hidden + 1)
        self.nodesO = [1.0] * out #no hidden node

        self.weightsI = self.generateWeights(self.nodesI,self.nodesH)
        self.weightsH = self.generateWeights(self.nodesH,self.nodesO)

        self.errorsI = [1.0] * (inp + 1)
        self.errorsH = [1.0] * (hidden + 1)
        self.errorsO = [1.0] * out #no hidden node
       # self.errorsI = [[0]*len(self.nodesH) for x in range(len(self.nodesI))]
       # self.errorsH = [[0]*len(self.nodesO) for x in range(len(self.nodesH))]

    def generateWeights(self,first,second):
        vector = list()
        for node in first:
            vector.append([random.uniform(-1, 1) for x in range(len(second))])
        return vector

    def train(self, examples):
        for example in examples:
            print("--------- new example --------")
            print example
            #input layer becomes input and the bias node is at 0
            self.nodesI = [self.nodesI[0], example[0],example[1]]

            #skip input Layer

            #skip bias node aka hl,0
            for hnode in range(1,len(self.nodesH)):
                sumation = 0
                #for every node in the input Layer
                for inNode in range(len(self.nodesI)):
                    #sum them up with their weight to the node in the hidden layer
                    weight =(self.weightsI[inNode][hnode]) 
                    sumation = sumation + self.nodesI[inNode] * weight
                self.nodesH[hnode] = sigmoid(sumation)

            #for every output node, no bias one
            for onode in range(0,len(self.nodesO)):
                sumation = 0
                #for every node in the hidden Layer
                for hidNode in range(len(self.nodesH)):
                    weight =(self.weightsH[hidNode][onode]) 
                    sumation = sumation + self.nodesH[hidNode] * weight
                self.nodesO[onode] = sigmoid(sumation)

            #calculate Error
            self.backProp(example)

    def backProp(self, example):
        for oNode in range(len(self.nodesO)):
            node = self.nodesO[oNode]
            self.errorsO[oNode] = node * (example[2] - node)
        for hNode in range(0,len(self.nodesH)):
            node = self.nodesH[hNode]
            errorSum = 0
            for oNode in range(len(self.nodesO)):
                errorSum = errorSum + (self.errorsO[oNode] * self.weightsH[hNode][oNode])
            errorSum = errorSum * self.nodesH[hNode] * (1 - self.nodesH[hNode])
            self.errorsH[hNode] = errorSum
        for iNode in range(0,len(self.nodesI)):
            node = self.nodesI[iNode]
            errorSum = 0
            for hNode in range(len(self.nodesH)):
                errorSum = errorSum + (self.errorsH[hNode] * self.weightsI[iNode][hNode])
            errorSum = errorSum * self.nodesI[iNode] * (1 - self.nodesI[iNode])
            self.errorsI[iNode] = errorSum
        
        # update the weights now
        for hNode in range(len(self.nodesH)):
            for oNode in range(len(self.nodesO)):
                error = 0.1 * self.errorsO[oNode] * self.nodesH[hNode]
                self.weightsH[hNode][oNode] =self.weightsH[hNode][oNode] - error

        for iNode in range(len(self.nodesI)):
            for hNode in range(len(self.nodesH)):
                error = 0.1 * self.errorsH[hNode] * self.nodesI[iNode]
                self.weightsI[iNode][hNode] =self.weightsI[iNode][hNode] - error

    def getError(self):
        return reduce(lambda x, y: x+(y**2), self.errorsO) + \
               reduce(lambda x, y: x+(y**2), self.errorsI)  + \
               reduce(lambda x, y: x+(y**2), self.errorsH)

def sigmoid(x):
    return (1/(1+(math.e**(-x))))

def sigDeriv(x):
    return (sigmoid(x)*(1-sigmoid(x)))

def main():
    nn = NeuralNet(2,5,4)
    print(nn)
    testVectors = getTrainingSet()
    nn.train(testVectors)
    print(nn.getError())
    print(nn)
    for epoch in range(0, 1):
        #nn.train(testVectors)

        if epoch in [10**x for x in range(5)]:
            #store pickle of NN
            output = open('nn'+str(epoch)+'.pkl', 'wb')
            pickle.dump(nn, output)
            output.close()
            
            
####

def getTrainingSet():
    random.seed(42)
    fin = open('train_data.csv')
    testVectors = []
    for line in fin:
        aList = line.strip().split(',')
        testVectors.append((float(aList[0]),float(aList[1]),int(aList[2])))
    return testVectors
main()

