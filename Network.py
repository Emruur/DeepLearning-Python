import numpy as np
import random

class Network:

    def __init__(self,sizes) -> None:
        self.layers= len(sizes)
        self.sizes= sizes
        weight_shapes= np.array([(a,b) for a,b in zip(sizes[1:],sizes[:-1])])
        self.weights= np.array([np.random.standard_normal(s) for s in weight_shapes],dtype=object)
        self.biases= np.array([np.zeros((s,1)) for s in sizes[1:]],dtype=object)

    def __init__(self,sizes,weights,biases) -> None:
        self.layers= len(sizes)
        self.sizes= sizes
        self.weights= weights
        self.biases= biases
        

    def feedForward(self,input):
        #takes an input vector and outputs a vector based on the current weights and biases

        a= input

        for w,b in zip(self.weights, self.biases):
            #applies matrix vector multiplication to every layer and adds the bias vector to it
            # where w is a matrix of weights and a,b is a vector(can be viewed as a n,1 matrix)
            #output of one is the input of the preceding layer

            a= Network.sigmoid(np.matmul(w,a)+b)
        return a
    
    def printWeights(self):
        for w in self.weights:
            print(w,"\n")

    def printBiases(self):
        for b in self.biases:
            print(b,"\n")

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    
    @staticmethod
    def sigmoid_prime(z):
        """Derivative of the sigmoid function."""
        return Network.sigmoid(z)*(1-Network.sigmoid(z))


    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        #for every epoch the whole trainibg set is split into mini batches after being randomized
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            #Every mini batch is fed as an input for gradient descent algorithm
            # Then for each mini_batch we apply a single step of gradient descent
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j+1, self.evaluate(test_data), n_test)) 
            else:
                print("Epoch "+str(j+1)+" complete...") 
        print("Training complete!")

    def update_mini_batch(self, mini_batch, eta):

        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        
        #Initializing arrrays of 0's in the shape of weights and biases
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # x= training input , y= expected output
        for x, y in mini_batch:
            #for every training input it computes a gradient vector(delta_nabla) 
            # for weights and biases and addes them to a vector so that it can be averaged after 
            #summing it up for all the mini batch
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        #Substracting the nabla vector from the weight vector since we are aiming for the steepest #
        # descent which is the negative of the gradient 
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b-(eta/len(mini_batch))*nb
                    for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #    feedForward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = Network.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            Network.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.    layers):
            z = zs[-l]
            sp = Network.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""

        return (output_activations-y)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""

        #Aimed at testing the progress, returns how many of the inputs in the whole test data
        #that the network guessed correctly

        test_results = np.array([(np.argmax(self.feedForward(x)), np.argmax(y))
                        for (x, y) in test_data])

        return sum(int(x==y) for (x,y) in test_results)


        


