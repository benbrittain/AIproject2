from pylab import *
import pickle
import random 
import math

class NeuralNet():
    def __str__(self):
        stri = "\n"
        stri = stri + "\nOutput Layer" + str(self.nodesO)
        stri = stri + "\nHidden Layer" + str(self.nodesH)
        stri = stri + "Input Layer" + str(self.nodesI)
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

        self.weightsI = self.generateWeights(len(self.nodesI),len(self.nodesH)-1)
        self.weightsH = self.generateWeights(len(self.nodesH),len(self.nodesO))

        self.errorsI = [1.0] * (inp + 1)
        self.errorsH = [1.0] * (hidden + 1)
        self.errorsO = [1.0] * out #no hidden node

    def generateWeights(self,first,second):
        vector = list()
        for node in range(first):
            vector.append([random.uniform(-1, 1) for x in range(second)])
        return vector

    def train(self, examples):
        for example in examples:
            #input layer becomes input and the bias node is at 0
            self.nodesI = [self.nodesI[0], example[0],example[1]]

            #skip input Layer

            #skip bias node aka hl,0
            for hnode in range(0,len(self.nodesH)-1):
                sumation = 0
                #for every node in the input Layer
                for inNode in range(len(self.nodesI)):
                    weight =(self.weightsI[inNode][hnode]) 
                    sumation = sumation + (self.nodesI[inNode] * weight)
                self.nodesH[hnode] = sigmoid(sumation)
                #print hnode, sumation, sigmoid(sumation)

            #for every output node, no bias one
            for onode in range(0,len(self.nodesO)):
                sumation = 0
                #for every node in the hidden Layer
                for hidNode in range(len(self.nodesH)):
                    weight =(self.weightsH[hidNode][onode]) 
                    sumation = sumation + self.nodesH[hidNode] * weight
                self.nodesO[onode] = sigmoid(sumation)

#            #calculate Error
            self.backProp(example)

    def backProp(self, example):
        for oNode in range(len(self.nodesO)):
            node = self.nodesO[oNode]
            value = 0
            if (oNode == (example[2]-1)):
                value = 1
            self.errorsO[oNode] = (node - value) * (node * (1-node))

        for hNode in range(0,len(self.nodesH)-1):
            node = self.nodesH[hNode]
            errorSum = 0
            for oNode in range(len(self.nodesO)):
#                print "adding error of output", oNode
#                print "adding error of output", self.errorsO[oNode]
                errorSum = errorSum +(self.weightsH[hNode][oNode]* self.errorsO[oNode])
            self.errorsH[hNode] = errorSum * (node)*(1-node)
#        print self.weightsH
#        print self.errorsH

        for iNode in range(0,len(self.nodesI)):
            node = self.nodesI[iNode]
            errorSum = 0
            #because it doesn't link to the bias node
            for hNode in range(len(self.nodesH)-1):
                errorSum = errorSum + self.weightsI[iNode][hNode]*self.errorsH[hNode]
            self.errorsI[iNode] = errorSum * (node)*(1-node)

        # update the weights now
        for hNode in range(len(self.nodesH)-1):
            for oNode in range(len(self.nodesO)):
                error = 0.1 * self.errorsO[oNode] * self.nodesH[hNode]
                self.weightsH[hNode][oNode] =self.weightsH[hNode][oNode] - error

        for iNode in range(1,len(self.nodesI)):
            for hNode in range(len(self.nodesH)-1):
                error = 0.1 * self.errorsH[hNode] * self.nodesI[iNode]
                self.weightsI[iNode][hNode] =self.weightsI[iNode][hNode] - error

    def getError(self):
        print "--------"
        print self.nodesO
        print self.errorsO
        return reduce(lambda x, y: x+(y**2), self.errorsO) + \
               reduce(lambda x, y: x+(y**2), self.errorsI)  + \
               reduce(lambda x, y: x+(y**2), self.errorsH)

    def classify(self, example):

        #put train logic in here

       current = float("-inf")
       classify = -1
       for oNode in range(0,len(self.nodesO)):
            if self.nodesO[oNode] > current:
              classify = oNode
              current = self.nodesO[oNode]
       return classify

def sigmoid(x):
        return 1 / (1 + math.exp(-x))


def getTrainingSet():
    fin = open('train_data.csv')
    testVectors = []
    for line in fin:
        aList = line.strip().split(',')
        testVectors.append((float(aList[0]),float(aList[1]),int(aList[2])))
    return testVectors

nn = pickle.load(open('nntest.pkl', 'rb'))
testVectors = getTrainingSet()
#print nn
nn.train(testVectors)
#nn.train(testVectors[0:1])

#
def main():
    nn = pickle.load(open('nntest.pkl', 'rb'))
    testVectors = getTrainingSet()

    for epoch in range(0, 1001):
        nn.train(testVectors)
        print(nn.getError())
        if epoch in [10**x for x in range(5)]:
            #store pickle of NN
            output = open('nn'+str(epoch)+'.pkl', 'wb')
            pickle.dump(nn, output)
            output.close()

main()
