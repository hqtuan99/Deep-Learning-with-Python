import numpy as np

class MLP:
    def __init__(self, num_inputs=3, num_hidden=[3, 3], num_outputs=2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        # create a generic representation of the layers
        layers = [num_inputs] + num_hidden + [num_outputs]
        # initiate random weights for the layers
        weights = []
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1]) # Create a 2D random array 
            weights.append(w)
        self.weights = weights
            
    # the input layer activation is just the input itself        
    def forward_propergate(self, inputs):
        activations = inputs
        
        # repeat through the network layers
        for i, w in enumerate(self.weights):
            # calculate the net inputs
            net_inputs = np.dot(activations, w) # Perform multiplication between activations and weights
            # calculate the activations using sigmoid function
            activations = self._sigmoid(net_inputs)
        return activations 
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
if __name__ == "__main__":
    # create an MLP
    mlp = MLP()
    
    # create some inputs
    inputs = np.random.rand(mlp.num_inputs)
    
    # perform forward prop
    outputs = mlp.forward_propergate(inputs)
    
    #print the results
    print("The network input is: {}".format(inputs))
    print("The network output is: {}".format(outputs))
    