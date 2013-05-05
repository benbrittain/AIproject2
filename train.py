import matplotlib.pyplot as plt
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

        self.weightsI = self.generateWeights(len(self.nodesI),len(self.nodesH))
        self.weightsH = self.generateWeights(len(self.nodesH),len(self.nodesO))

        self.errorsI = [1.0] * (inp+1)
        self.errorsH = [1.0] * (hidden+1)
        self.errorsO = [1.0] * out #no hidden node

    def generateWeights(self,first,second):
        vector = list()
        for node in range(first):
            vector.append([random.uniform(0, 1) for x in range(second)])
        return vector

    def train(self, examples):
        for example in examples:
            #input layer becomes input and the bias node is at 0
            self.nodesI = [example[0],example[1],self.nodesI[2]]

            #skip bias node aka hl,0
            for hnode in range(len(self.nodesH)-1):
                sumation = 0
                #for every node in the input Layer
                for inNode in range(len(self.nodesI)):
                    weight = (self.weightsI[inNode][hnode]) 
                    sumation = sumation + (self.nodesI[inNode] * weight)
                self.nodesH[hnode] = sigmoid(sumation)
                #print hnode, sumation, sigmoid(sumation)

            #for every output node, no bias one
            for onode in range(len(self.nodesO)):
                sumation = 0
                #for every node in the hidden Layer
                for hidNode in range(len(self.nodesH)):
                    weight =(self.weightsH[hidNode][onode]) 
                    sumation = sumation + (self.nodesH[hidNode] * weight)
                self.nodesO[onode] = sigmoid(sumation)

            self.backProp(example)

    def backProp(self, example):

        for oNode in range(len(self.nodesO)):
            node = self.nodesO[oNode]
            if (oNode == (example[2]-1)):
                value = 1
            else:
                value = 0
            self.errorsO[oNode] =  (value - node)*(node * (1-node)) 

        for hNode in range(len(self.nodesH)):
            node = self.nodesH[hNode]
            errorSum = 0
            for oNode in range(len(self.nodesO)):
                errorSum = errorSum + (self.weightsH[hNode][oNode] * self.errorsO[oNode])
            self.errorsH[hNode] = errorSum * ((node)*(1-node))

        for iNode in range(len(self.nodesI)):
            node = self.nodesI[iNode]
            errorSum = 0
            for hNode in range(len(self.nodesH)-1):
                errorSum = errorSum + (self.weightsI[iNode][hNode] * self.errorsH[hNode])
            self.errorsI[iNode] = errorSum * ((node)*(1-node))

        # update the weights now
        for hNode in range(len(self.nodesH)):
            for oNode in range(len(self.nodesO)):
                error = 0.1 * self.errorsO[oNode] * self.nodesH[hNode]
                self.weightsH[hNode][oNode] = self.weightsH[hNode][oNode] + error

        for iNode in range(len(self.nodesI)):
            for hNode in range(len(self.nodesH)):
                error = 0.1 * self.errorsH[hNode] * self.nodesI[iNode]
                self.weightsI[iNode][hNode] = self.weightsI[iNode][hNode] + error

        count = 0
        maxNode = float("-inf")
        for x in range(len(self.nodesO)):
            node = self.nodesO[x]
            if node > maxNode:
                maxNode = node
                count = x

#        print count == (example[2]-1)

    def getError(self):
        return reduce(lambda x, y: x+(y**2),self.errorsO, 0) + \
               reduce(lambda x, y: x+(y**2),self.errorsI, 0)  + \
               reduce(lambda x, y: x+(y**2),self.errorsH, 0)

    def classify(self, ecc, sym):
        self.nodesI = [ecc,sym,self.nodesI[2]]

        #skip bias node aka hl,0
        for hnode in range(len(self.nodesH)-1):
            sumation = 0
            for inNode in range(len(self.nodesI)):
                weight = (self.weightsI[inNode][hnode]) 
                sumation = sumation + (self.nodesI[inNode] * weight)
            self.nodesH[hnode] = sigmoid(sumation)

        for onode in range(len(self.nodesO)):
            sumation = 0
            for hidNode in range(len(self.nodesH)):
                weight =(self.weightsH[hidNode][onode]) 
                sumation = sumation + (self.nodesH[hidNode] * weight)
            self.nodesO[onode] = sigmoid(sumation)

        count = 0
        maxNode = float("-inf")
        for x in range(len(self.nodesO)):
            node = self.nodesO[x]
            if node > maxNode:
                maxNode = node
                count = x
        return (count+1)

def sigmoid(x):
        return 1 / (1 + math.exp(-x))

def getTrainingSet():
    fin = open('train_data.csv')
    testVectors = []
    for line in fin:
        aList = line.strip().split(',')
        testVectors.append((float(aList[0]),float(aList[1]),int(aList[2])))
    return testVectors

#nn = NeuralNet(2,5,4)
#testVectors = getTrainingSet()
#nn.train(testVectors)


def main():
    nn = NeuralNet(2,5,4)
    testVectors = getTrainingSet()
    epochs = []
    points = []
    
    for epoch in range(0, 100001):
        nn.train(testVectors)
        epochs.append(epoch)
        points.append(nn.getError())
    #    if epoch == 10000:
        if epoch in [10**x for x in range(5)]:
            print("Epoch "+str(epoch)+": " + str(nn.getError()))

            plt.xlabel('Epoch')
            plt.ylabel('SSD')
            plt.title('less stupid network')
            plt.plot(epochs,points)
            plt.show()

            #store pickle of NN
            output = open('nn'+str(epoch)+'.pkl', 'wb')
            pickle.dump(nn, output)
            output.close()

if __name__ == '__main__':
    main()
