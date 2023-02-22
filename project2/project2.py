############################################
# Authors: Tanner Mengel and Ian Cox
# Date: 2/28/2021
############################################
# Imports
import numpy as np
import sys

class Neuron:
    '''
    Class for a single neuron in a neural network
    '''
    def __init__(self, input_dim, activation, learning_rate=0.1, weights=None, verbose=False):
        # Initialize the neuron with the following parameters
        self.input_dim = input_dim # Number of inputs plus 1 for bias (set eariler)
        self.activation = activation  # Activation function
        # Activation function can be either 'linear' or 'logistic'
        if activation not in ['linear', 'logistic']:
            raise ValueError(f'Activation function {activation} not supported')
    
        self.learning_rate = learning_rate # Learning rate
        # Weights are initialized randomly if not provided
        if weights is None: self.weights = np.random.rand(input_dim)
        else: self.weights = weights
        if self.weights.shape[0] != input_dim:
            raise ValueError(f'Input dim does not match the dim of weights, input dim: {input_dim}, weights dim: {self.weights.shape[0]}')
        # Initialize the output and inputs to None
        self.output = None
        self.inputs = []
        # Initialize the partial derivatives to None
        self.dW = []
        
        if verbose: # Print the parameters if verbose is True
            print('Initialized Neuron with the following architecture:')
            print(f'Input Dim: {input_dim}, Activation Function: {activation}')
            print(f'Learning Rate: {learning_rate}, Weights: {weights}')
            print(f'Weights Shape: {weights.shape}')
    
    # This method returns the output of the neuron given the input
    def active(self, input): # Activation function
        if self.activation == 'linear': return input
        elif self.activation == 'logistic': return 1/(1+np.exp(-input))
        else: raise ValueError(f'Activation function {self.activation} not supported')

    # This method calculates the output of the neuron given the input
    def calculate(self, X):
        self.inputs = X
        input = np.dot(self.weights, X) # Calculate the dot product of the weights and the inputs
        output = self.active(input) # Calculate the output of the neuron after applying the activation function
        self.output = output
        return output # Return the output of the neuron
    
    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):
        if self.activation == "linear": return 1
        elif self.activation == "logistic": return self.output*(1-self.output) 
        else : raise ValueError("Activation function not found")
      
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        # print('calcpartialderivative') 
        delta = wtimesdelta*self.activationderivative()
        self.dW = self.inputs*delta #Calculate the partial derivative for each weight
        return delta*self.weights
               
    #Simply update the weights using the partial derivatives and the leranring weight
    def updateweight(self):
        for i in range(len(self.weights)): #Update the weights
            self.weights[i] -= self.learning_rate*self.dW[i]
    
class FullyConnectedLayer:
    '''
    Class for a single layer in a neural network to manage the neurons in the layer
    '''
    def __init__(self, neuron_num, activation, input_num, learning_rate=0.1, weights=None, verbose=False):
        # Initialize the layer with the following parameters
        self.neuron_num = neuron_num   # Number of neurons in the layer
        self.activation = activation  # Activation function
        # Activation function can be either 'linear' or 'logistic'
        self.learning_rate = learning_rate # Learning rate
        self.input_num = input_num # Number of inputs to the layer (including bias)
        # Weights are initialized randomly if not provided
        if weights is None:
            weights = []
            for i in range(neuron_num): 
                weights.append(np.random.rand(input_num)) # Initialize the weights randomly
        self.weights = weights
        # Initialize the neurons in the layer
        self.neurons = []
        for i in range(neuron_num):
            weight = weights[i]
            inputdim = weight.shape[0]
            if inputdim != input_num:# Check if the input dimension matches the dimension of the weights
                raise ValueError(f'Input dim does not match the dim of weights, input dim: {inputdim}, weights dim: {input_num}')
            self.neurons.append(Neuron(input_dim=inputdim, activation=activation, learning_rate=learning_rate, weights=weight))
        # Initialize the output and inputs to None
        self.outputs = []
        if verbose: # Print the parameters if verbose is True
            print('Initialized Layer with the following architecture:')
            print(f'Neuron Number: {neuron_num}, Activation Function: {activation}')
            print(f'Input Number: {input_num}, Learning Rate: {learning_rate}')
            print(f'Weights: {weights}')
            print(f'Weights Length: {len(weights)}')
    # This method calculates the output of the layer given the input
    def calculate(self, X):
        input = np.append(X,1)
        output = []
        for neuron in self.neurons:
            output.append(neuron.calculate(input)) # Calculate the output of each neuron in the layer
        return output
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcwdeltas(self, dloss):
        #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its own w*delta, and then update the weights (using the updateweight() method). I should return the sum of w*delta.          
        deltaw = np.zeros(self.input_num)
        for i in range(self.neuron_num):
            deltaw += self.neurons[i].calcpartialderivative(dloss[i]) #Calculate the partial derivative sum for each weight
            self.neurons[i].updateweight()    #Update the weights
        return deltaw #Return the sum of w*delta
    
class ConvolutionalLayer:
    '''
    Convoluational Layer Class to manage the neurons in the layer
    '''
    def __init__(self, num_kernels, kernel_size, activation, input_dim, learning_rate, weights=None):
        print('Convolutional Layer')
        self.number_of_neurons = (input_dim[0]-kernel_size+1)*(input_dim[1]-kernel_size+1)*num_kernels
        self.num_kernels = num_kernels
        self.activation = activation
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.weights = weights
        self.output_shape= (self.input_dim[0]-self.kernel_size+1, self.input_dim[1]-self.kernel_size+1, self.num_kernels)
        if weights is None:
            self.weights = np.random.rand(num_kernels, input_dim[2]*kernel_size*kernel_size+1)
        self.outputs = []
        self.neurons = []
        for i in range(num_kernels):
            weight = self.weights[i]
            inputdim = weight.shape[-1]
            num_neurons = (input_dim[0]-kernel_size+1)*(input_dim[1]-kernel_size+1)
            for j in range(num_neurons):
                self.neurons.append(Neuron(input_dim=inputdim, activation=activation, learning_rate=learning_rate, weights=weight))
        
    def calculate(self, input):
        self.outputs = np.zeros((self.input_dim[0]-self.kernel_size+1, self.input_dim[1]-self.kernel_size+1, self.num_kernels))
        if input.shape != tuple(self.input_dim):
            raise ValueError(f'Input shape does not match the input dim, input shape: {input.shape}, input dim: {self.input_dim}')
        #print(self.num_kernels)
        for i in range(self.num_kernels):
           # print(self.input_dim[0],self.kernel_size)
            for j in range(self.input_dim[0]-self.kernel_size+1):
                #print(self.input_dim[1]-self.kernel_size+1)
                for k in range(self.input_dim[1]-self.kernel_size+1):
                    input_patch = input[j:j+self.kernel_size, k:k+self.kernel_size, :]
                    #print("Input patch" ,input_patch.shape)
                    input_patch = input_patch.reshape(-1)
                    input_patch = np.append(input_patch, 1)
                    self.outputs[j, k, i] = self.neurons[i*(self.input_dim[0]-self.kernel_size+1)*(self.input_dim[1]-self.kernel_size+1)+j*(self.input_dim[1]-self.kernel_size+1)+k].calculate(input_patch)
        return self.outputs
    
    def calculatewdeltas(self, next_delta):
        print('Calculate w deltas')
        #deltaw = np.zeros(self.weights.shape)
        
class MaxPoolingLayer:
    '''
    Max Pooling Layer
    '''
    def __init__(self, kernel_size, input_shape):
        print('Max Pooling Layer')
        print(input_shape, kernel_size)
        self.number_of_neurons = (input_shape[0]-kernel_size+1)*(input_shape[1]-kernel_size+1)
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.outputs = []
        self.xPool = (self.input_shape[0]-self.kernel_size+1)
        self.yPool = (self.input_shape[1]-self.kernel_size+1)
        self.output_shape = (self.xPool, self.yPool)
        
    def calculate(self, input):
        # print(input)
        print('Calculate Max Pooling')
        for k in range(self.input_shape[2]):
            h = []
            for j in range(self.xPool):
                l = []
                for i in range(self.yPool):
                    l.append(np.amax(input[j:j+self.kernel_size,i:i+self.kernel_size,k]))
                h.append(l)
            self.outputs.append(h)
        return np.array(self.outputs)

    
    def calculatewdeltas(self, next_layer_wdeltas):
        print('Calculate w deltas')
       
class FlattenLayer:
    '''
    Flattening Layer
    '''
    def __init__(self,input_shape):
        print('Flatten Layer')
        self.outputs = None
        self.inputs = None
        self.input_shape = input_shape
        self.output_shape = np.prod(input_shape)
        
    def calculate(self, input):
        print('Calculate Flatten')
        self.inputs = input
        self.outputs = input.flatten()
        return self.outputs
        
class NeuralNetwork:
    '''
    Class for a neural network to manage the layers and the training
    '''        

    def __init__(self, input_size, loss, learning_rate=0.1,verbose=False):
        # Initialize the neural network with the following parameters
        self.num_of_layers = 0 # Number of layers in the neural network
        if type(input_size) is not int: # Check if the input size is an integer
            input_size.append(1)
            
        self.input_size = np.array(input_size) # Number of inputs to the neural network
        if loss not in ['mse', 'bce']: # Check if the loss function is supported
            raise ValueError(f'Loss function {loss} not supported')
        self.loss = loss # Loss function
        self.learning_rate = learning_rate # Learning rate
        self.architecture = [input_size]  # Architecture of the neural network
        self.num_of_neurons = [] # Number of neurons in each layer
        self.layer_types = [] # Type of each layer
        self.activation = [] # Activation function for each layer
        self.weights = [] # Weights for each layer
        self.layers = []  # Layers in the neural network
        self.output = [] # Outputs of the neural network
        
        if verbose: # Print the architecture of the neural network
            print('Initialized Neural Network with the following architecture:')
            print(f'Input Size: {input_size}, Loss: {loss}, Learning Rate: {learning_rate}')
    
    def addLayer(self, layer_type, num_of_neurons=0, activation="linear", kernel_size=1, weights=None):
        self.num_of_layers += 1
        self.num_of_neurons.append(num_of_neurons)
        self.layer_types.append(layer_type)
        if activation not in ['linear', 'logistic']:
            raise ValueError(f'Activation function {activation} not supported')
        self.activation.append(activation)
        
        # Add the layer to the neural network
        if layer_type == 'fullyconnected':
            print('Adding Fully Connected Layer')
            self.layers.append(FullyConnectedLayer(num_of_neurons, activation,self.architecture[-1]+1, self.learning_rate, weights))
            self.architecture.append(num_of_neurons)
        elif layer_type == 'convolutional':
            self.layers.append(ConvolutionalLayer(num_of_neurons, kernel_size, activation, self.architecture[-1], self.learning_rate, weights))
            self.architecture.append(self.layers[-1].output_shape)
            print('Adding Convolutional Layer')
        elif layer_type == 'maxpooling':
            print(self.architecture)
            self.layers.append(MaxPoolingLayer(kernel_size,self.architecture[-1]))
            self.architecture.append(self.layers[-1].output_shape)
            print('Adding Max Pooling Layer')
        elif layer_type == 'flatten':
            self.layers.append(FlattenLayer(self.architecture[-1]))
            self.architecture.append(self.layers[-1].output_shape)
            print('Adding Flatten Layer')
        else:
            raise ValueError(f'Layer type {layer_type} not supported')
       
            
     # Calculate the output of the neural network 
    def calculate(self,X):
        input = np.array(X)
        #print(input.shape)
        #print(input[0])
        if self.input_size.shape != input.shape: # Check if the input size is an integer
           input = input.reshape(self.input_size)
        #print(input[0])
        # return
        for layer in self.layers:
            # input = np.append(input, 1) # Append a 1 to the input to account for the bias
            output = layer.calculate(input)# Calculate the output of the layer
            input = output
        return output   # Return the output of the neural network
    
    # Calculate the loss of the neural network with respect to the ground truth
    def calculateloss(self,yhat,y):
        if self.loss == 'mse':  # Calculate the mean squared error
            loss = 0
            for i in range(len(yhat)):
                loss += (yhat[i] - y[i])**2
            loss = loss/len(yhat) # Calculate the mean
            return loss
        elif self.loss == 'bce':   # Calculate the binary cross entropy
            loss = 0
            for i in range(len(yhat)):
                loss += -y[i]*np.log(yhat[i]) - (1-y[i])*np.log(1-yhat[i])
            loss = loss/len(yhat) # Calculate the mean
            return loss
        else : raise ValueError(f'Loss function {self.loss} not supported') # Raise an error if the loss function is not supported
    
    # Calculate the derivative of the loss function with respect to the output of the neural network
    def lossderiv(self,yhat,y):
        if len(yhat)==1:# If the output is a scalar, return a list
            if self.loss == 'mse':  # Calculate the derivative of the mean squared error
                dloss = 0
                for i in range(len(yhat)):
                    dloss = 2*(yhat[i] - y[i])
                dloss = dloss/len(yhat) # Calculate the mean
                return [dloss]
            elif self.loss == 'bce': # Calculate the derivative of the binary cross entropy
                dloss = 0
                for i in range(len(yhat)):
                    dloss = -y[i]/yhat[i] + (1-y[i])/(1-yhat[i])
                dloss = dloss/len(yhat) # Calculate the mean
                return [dloss]
            else : raise ValueError(f'Loss function {self.loss} not supported') # Raise an error if the loss function is not supported
        else: # If the output is a vector, return a vector
            dloss = []# Initialize the derivative of the loss
            if self.loss == 'mse': # Calculate the derivative of the mean squared error
                for i in range(len(yhat)):
                    dloss.append(2*(yhat[i] - y[i]))
                dloss = np.array(dloss)/len(yhat)
                return dloss
            elif self.loss == 'bce': # Calculate the derivative of the binary cross entropy
                for i in range(len(yhat)):
                    dloss.append(-y[i]/yhat[i] + (1-y[i])/(1-yhat[i]))
                dloss = np.array(dloss)/len(yhat)
                return dloss
            else : raise ValueError(f'Loss function {self.loss} not supported')
    
    # Train the neural network with the given inputs and outputs
    def train(self, X, Y, epochs=1, verbose=False):
        
        for epoch in range(epochs): # Iterate through the epochs
            loss = 0 # Initialize the loss
            yhat = [] # Initialize the output of the neural network
            ytest = [] # Initialize the ground truth
            for i in range(len(X)): # Iterate through the training examples
                yh = self.calculate(X[i])     # Calculate the output of the neural network
                dloss = self.lossderiv(yh, Y[i]) # Calculate the derivative of the loss function with respect to the output of the neural network
                for j in range(len(self.layers)-1, -1, -1): # Iterate through the layers in reverse order
                    dloss = self.layers[j].calcwdeltas(dloss)  # Calculate the derivative of the loss function with respect to the weights of the layer  
                yhat.append(self.calculate(X[i])) # Calculate the output of the neural network
                ytest.append(Y[i]) # Calculate the ground truth
                loss += self.calculateloss(yhat[i], Y[i])/len(X)
            if epoch % 10000 == 0: # Print the loss every 10000 epochs
                print(f'Epoch: {epoch}, Loss: {loss}') # Print the loss
         
if __name__=="__main__": # Run the main function
    learningRate = float(sys.argv[1])
    print('Using a learning rate of',learningRate)
    if (len(sys.argv)<3):
        print('a good place to test different parts of your code')
    
    if(sys.argv[2] == 'CNN'):
        testCNN = NeuralNetwork(input_size=(28,28,3), loss='bce', learning_rate=0.01, verbose=True)
        testCNN.addLayer('convolutional', 7, 'logistic', kernel_size=5)
        test_input = np.random.rand(28,28,3)
        output = testCNN.calculate(test_input)
        output.shape