import numpy as np

class Network:

    def __init__(self,sizes) -> None:
        self.layers= len(sizes)
        self.sizes= sizes
        self.weight_shapes= np.array([(a,b) for a,b in zip(sizes[1:],sizes[:-1])])
        self.weights= np.array([np.random.standard_normal(s) for s in self.weight_shapes])
        self.biases= np.array([np.zeros((s,1)) for s in sizes[1:]])
        
    def printWightShapes(self):
        print(self.weight_shapes)

    def printWeights(self):
        for w in self.weights:
            print(w,"\n")

    def printBiases(self):
        for b in self.biases:
            print(b,"\n")
        

nw= Network([2,4,3])

nw.printBiases()


