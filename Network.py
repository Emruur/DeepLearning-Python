import numpy as np

class Network:

    def __init__(self,sizes) -> None:
        self.layers= len(sizes)
        self.sizes= sizes
        self.weight_shapes= np.array([(a,b) for a,b in zip(sizes[1:],sizes[:-1])])
        self.weights= np.array([np.random.standard_normal(s) for s in self.weight_shapes])
        self.biases= np.array([np.zeros((s,1)) for s in sizes[1:]])


    def feedForward(self,input):
        #takes an input vector and outputs a vector based on the current weights and biases

        a= input

        for w,b in zip(self.weights, self.biases):
            print(w,"\n\n",b,"\n","-------------------","\n")

        
    def printWightShapes(self):
        print(self.weight_shapes)

    def printWeights(self):
        for w in self.weights:
            print(w,"\n")

    def printBiases(self):
        for b in self.biases:
            print(b,"\n")

    def sigmoid(x):
        return 1/(np.exp(-x))
        

nw= Network([2,4,3])

nw.feedForward([])


