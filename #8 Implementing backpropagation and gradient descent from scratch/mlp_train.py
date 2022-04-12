import numpy as np
from random import random

# save activations and derivatives
# implement backpropagation
# implement gradient descent
# implement train
# train our net with some dummy dataset
# make som predictions

class MLP:
    def __init__(self, num_inputs=3, hidden_layers=[3, 3], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_hidden = hidden_layers
        self.num_outputs = num_outputs
        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]
        
        # initiate random weights for the layers
        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1]) # Create a 2D random array 
            weights.append(w)
        self.weights = weights
        
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations
        
        derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            derivatives.append(d)
        self.derivatives = derivatives
            
            
    def forward_propergate(self, inputs):
        # the input layer activation is just the input itself
        activations = inputs
        
        # save the activation for backpropagation
        self.activations[0] = activations
        
        # repeat through the network layers
        for i, w in enumerate(self.weights):
            # calculate the net inputs
            net_inputs = np.dot(activations, w) # Perform multiplication between activations and weights
            # calculate the activations using sigmoid function
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations
        return activations 
    
    def back_propagate(self, error, verbose=False): 
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivatives(activations) # ndarray([0.1, 0.2]) --> ndarray([[0.1, 0.2]]) / Reshape to 2D array with single raw
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i] # ndarray([0.1, 0.2]) --> ndarray([[0.1], [0.2]]) / Reshape to vertical vector
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)    
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)
            
            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))
                
        return error
            
    def _sigmoid_derivatives(self, x):
        return x * (1.0 - x)
    
    def _sigmoid(self, x):
        y = 1 / (1 + np.exp(-x))
        return y
if __name__ == "__main__":
    # create an MLP
    mlp = MLP(2, [5], 1)
    
    # create dummy data
    inputs = np.array([0.1, 0.2])
    target = np.array([0.3])
    
    # forward propagation
    output = mlp.forward_propergate(inputs) 
    
    # calculate error
    error = target - output
    
    # back propagation
    mlp.back_propagate(error, verbose=True)
    