import numpy as np
import Network as nn

with np.load('mnist.npz') as data: 
	training_images = data['training_images']
	training_labels= data["training_labels"]

network= nn.Network((784,16,16,10))

n_out = network.feedForward(training_images)

print(n_out[1],"\n",training_labels[1])


