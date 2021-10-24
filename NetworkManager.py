import numpy as np
import Network as nn

with np.load('mnist.npz') as data: 
	training_images = data['training_images']
	training_labels= data["training_labels"]
	test_images= data["test_images"]
	test_labels= data["test_labels"]

training_set= [(x,y) for x,y in zip(training_images,training_labels)]
test_set= [(x,y) for x,y in zip(test_images,test_labels)]

def save_network(network)->None:
	'''Saves the weights and biases to a local file as weights,biases,sizes'''
	np.savez("wb.npz", weights=network.weights, biases=network.biases, sizes= network.sizes)

def create_network():
	'''Initializes a new neural network and returns it'''
	return nn.Network((786,16,16,10))

def train_network(network, epochs= 10, t_rate=1):
	'''Trains and returns the network with the available data imported'''
	network.SGD(training_set,epochs,10,t_rate, test_data=test_set)
	return network

def load_network():
	'''Loads and returns a locally stored neural network'''
	try:
		data= np.load("wb.npz")

		w= data["weights"]
		b= data["biases"]
		s= data["sizes"]

		return nn.Network(s,w,b)
	except:
		print("No network saved, create a new network!")
	










