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
        # Creating the specified number of neurons in the layer
        for i in range(neuron_num):
            weight = weights[i]
            inputdim = weight.shape[0]
            if inputdim != input_num:# Check if the input dimension matches the dimension of the weights
                raise ValueError(f'Input dim does not match the dim of weights, input dim: {inputdim}, weights dim: {input_num}')
            self.neurons.append(Neuron(input_dim=inputdim, activation=activation, learning_rate=learning_rate, weights=weight))
        # Initialize the output to None
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
        return np.array(output)
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcwdeltas(self, dloss):
        #given the next layer's w*delta, run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its own w*delta, and then update the weights (using the updateweight() method). I return the sum of w*delta.          
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
        # Initializing the parameters in the ConvolutionalLayer Class
        self.number_of_neurons = (input_dim[0]-kernel_size+1)*(input_dim[1]-kernel_size+1)*num_kernels # Each neuron is used for calculating the kernel onto a specific portion of the input
        self.num_kernels = num_kernels # Also known as the number of filters
        self.activation = activation
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.weights = weights
        self.output_shape= (self.input_dim[0]-self.kernel_size+1, self.input_dim[1]-self.kernel_size+1, self.num_kernels)
        # the "pool" variables correspond to the output dimensions
        self.xPool = self.output_shape[0]
        self.yPool = self.output_shape[1]
        # Initializing random weights if not provided
        if weights is None:
            self.weights = np.random.rand(num_kernels, input_dim[2]*kernel_size*kernel_size+1)
        self.outputs = []
        
        self.neurons = []
        # Creating the neurons in the layer
        # Have to loop over the 2 dimensions of the convolution
        for i in range(num_kernels):
            weight = None
            if weights is None:
                weight = self.weights[i]
                weight = np.append(weight.flatten(),0)
            else:
                weight = self.weights[i]
            inputdim = weight.shape[-1]
            num_neurons = (input_dim[0]-kernel_size+1)*(input_dim[1]-kernel_size+1)
            for j in range(num_neurons):
                self.neurons.append(Neuron(input_dim=inputdim, activation=activation, learning_rate=learning_rate, weights=weight))
        
    def calculate(self, input):
        # Calculating the output of the Convolutional layer with its own weights acting on the input passed to it
        self.outputs = np.zeros((self.xPool, self.yPool, self.num_kernels))
        if input.shape != tuple(self.input_dim):
            raise ValueError(f'Input shape does not match the input dim, input shape: {input.shape}, input dim: {self.input_dim}')
        for i in range(self.num_kernels):
            for j in range(self.xPool):
                for k in range(self.yPool):
                    input_patch = input[j:j+self.kernel_size, k:k+self.kernel_size, :]
                    input_patch = input_patch.reshape(-1)
                    input_patch = np.append(input_patch, 1)
                    self.outputs[j, k, i] = self.neurons[i*(self.xPool)*(self.yPool)+j*(self.yPool)+k].calculate(input_patch)
        return self.outputs
    
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcwdeltas(self, dloss):
        dLossdOut = np.zeros(self.input_dim)
        dloss = np.array(dloss)
        for i in range(self.num_kernels):
            for j in range(self.xPool):
                for k in range(self.yPool):
                    print()
                    dLossdOut[j, k, :] += np.sum(self.neurons[i*(self.xPool)*(self.yPool)+j*(self.yPool)+k].calcpartialderivative(dloss[j,k,i])) # Calculating the partial derivative sum for each weight
                    self.neurons[i*(self.xPool)*(self.yPool)+j*(self.yPool)+k].updateweight()
        return dLossdOut
               
class MaxPoolingLayer:
    '''
    Max Pooling Layer
    '''
    def __init__(self, kernel_size, input_shape,padding='valid'):
        # Initializing the MaxPoolingLayer Class
        self.number_of_neurons = ((input_shape[0]-kernel_size+1)*(input_shape[1]-kernel_size+1))/kernel_size
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.outputs = []
        self.stride = kernel_size
        # These are the number of "pools" which will occur, and incidentally is also the output dimensions
        self.xPool = ((self.input_shape[0]-self.kernel_size)//self.stride+1)
        self.yPool = ((self.input_shape[1]-self.kernel_size)//self.stride+1)
        self.output_shape = (self.xPool, self.yPool,input_shape[2])
        # This array is used as the mask telling which location in the input corresponded to the maximum value in the pool.  Needed to find where to pass back the dloss
        self.maxArgs = np.zeros(input_shape)
        
    def calculate(self, input):
        # need to reinitialize the outputs before computing
        self.maxArgs = np.zeros(self.input_shape)
        self.outputs = []
        # Finding the maximum in a subset (pool) of the input, looping over all of the subsets which constitute the input
        # Also creating the mask used in back propagation
        for k in range(self.input_shape[2]):
            h = []
            for j in range(self.xPool):
                l = []
                for i in range(self.yPool):
                    Maxl = np.amax(input[j*self.stride:j*self.stride+self.kernel_size,i*self.stride:i*self.stride+self.kernel_size,k])
                    MaxlArg = np.argmax(input[j*self.stride:j*self.stride+self.kernel_size,i*self.stride:i*self.stride+self.kernel_size,k])
                    MaxArgX_Rel = MaxlArg//self.kernel_size
                    MaxlArgY_Rel = MaxlArg%self.kernel_size
                    self.maxArgs[j*self.stride+MaxArgX_Rel,i*self.stride+MaxlArgY_Rel,k] = 1
                    l.append(Maxl)
                h.append(l)
            self.outputs.append(h)
        return np.array(self.outputs)

    # Returns the delta*w corresponding to the maximum values in the pools, where the location of the values are saved in the mask maxArgs
    def calcwdeltas(self, dloss):
        dloss = np.array(dloss)
        deltaw = np.zeros((self.input_shape[0], self.input_shape[1],self.input_shape[2]))
        for k in range(self.input_shape[2]):
            for j in range(self.xPool):
                for i in range(self.yPool):
                    self.maxArgs[j*self.stride:j*self.stride+self.kernel_size,i*self.stride:i*self.stride+self.kernel_size,k]*=dloss[j,i,k]
        deltaw += self.maxArgs
        return deltaw            
         
class FlattenLayer:
    '''
    Flattening Layer
    '''
    def __init__(self,input_shape):
        # Initializing the variables in the FlattenLayer Class
        self.outputs = None
        self.inputs = None
        self.input_shape = input_shape
        self.output_shape = np.prod(input_shape)
        
    # This method flattens the input into a single-dimension array
    def calculate(self, input):
        self.inputs = input
        self.outputs = input.flatten()
        return self.outputs
    
    # This method reshapes the dloss from a previous layer into the shape of the input layer for back propagation
    def calcwdeltas(self,dloss):
        dloss = dloss[:-1]
        deltaw = np.reshape(dloss, self.input_shape)
        return deltaw
        
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
        if self.input_size.shape != input.shape: # Check if the input size is an integer
           input = input.reshape(self.input_size)
        for layer in self.layers:
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
        X=np.array(X)
        Y=np.array(Y)
        for epoch in range(epochs): # Iterate through the epochs
            loss = 0 # Initialize the loss
            yhat = [] # Initialize the output of the neural network
            ytest = [] # Initialize the ground truth
            for i in range(len(X)): # Iterate through the training examples
                yh = self.calculate(X[i])     # Calculate the output of the neural network
                # print('Output',yh)
                dloss = self.lossderiv(yh, Y[i]) # Calculate the derivative of the loss function with respect to the output of the neural network
                for j in range(len(self.layers)-1, -1, -1): # Iterate through the layers in reverse order
                    # print('Layer',j)
                    dloss = self.layers[j].calcwdeltas(dloss)  # Calculate the derivative of the loss function with respect to the weights of the layer  
                # yhat.append(self.calculate(X[i])) # Calculate the output of the neural network
                yhat.append(yh)
                ytest.append(Y[i]) # Calculate the ground truth
                loss += self.calculateloss(yhat[i], Y[i])/len(X)
            if epoch % 10000 == 0: # Print the loss every 10000 epochs
                print(f'Epoch: {epoch}, Loss: {loss}') # Print the loss
         
if __name__=="__main__": # Run the main function        
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras import layers
    from tensorflow.keras import optimizers

    if (len(sys.argv)<3):
        print('a good place to test different parts of your code')
    
    learningRate = float(sys.argv[1])
    print('Using a learning rate of',learningRate)

    if sys.argv[2] == 'example1':
        from parameters import generateExample1
        
        print('Example 1')
        # Call weight/data generating function
        l1k1,l1b1,l3,l3b,input, output = generateExample1()
        w1k1 = np.append(l1k1.flatten(),l1b1)
        w1 = np.array([w1k1])
        w3 = np.append(l3.flatten(),l3b)
        w3 = np.array([w3])
        
        #save initial values
        testCNNw1k1 = w1k1
        testCNNw3 = w3
        
        # Creating the custom Neural Network
        testCNN = NeuralNetwork(input_size=[5,5], loss='mse', learning_rate=learningRate, verbose=True)
        testCNN.addLayer('convolutional',1,'logistic',kernel_size=3,weights=w1)
        testCNN.addLayer('flatten',0,'logistic')
        testCNN.addLayer('fullyconnected',1,'logistic',weights=w3)

        #Create a feed forward network
        model=Sequential()
        # Add convolutional layers, flatten, and fully connected layer
        model.add(layers.Conv2D(1,3,input_shape=(5,5,1),activation='sigmoid')) 
        model.add(layers.Flatten())
        model.add(layers.Dense(1,activation='sigmoid'))
        # Call weight/data generating function
        #setting weights and bias of first layer.
        l1k1=l1k1.reshape(3,3,1,1)
        w1=l1k1
       
        tfCNNw1k1 = l1k1
        tfCNNw1b1 = l1b1[0]
        tfCNNw3 = np.transpose(l3)
        tfCNNb3 = l3b
        
        model.layers[0].set_weights([w1,np.array([l1b1[0]])]) #Shape of weight matrix is (w,h,input_channels,kernels)
        model.layers[2].set_weights([np.transpose(l3),l3b])
        
        #Setting input. Tensor flow is expecting a 4d array since the first dimension is the batch size (here we set it to one), and third dimension is channels
        img=np.expand_dims(input,axis=(0,3))
    
        sgd = optimizers.SGD(learning_rate=learningRate)
        model.compile(loss='MSE', optimizer=sgd, metrics=['accuracy'])
        history=model.train_on_batch(img,output)
        output_tf = model.predict(img)
        
        testCNN.train([input],[output],epochs=1)      
        output_custom = testCNN.calculate(input)  
        
        np.set_printoptions(precision=5)
        # print("Initial Comparision")
        # print('1st convolutional layer, 1st kernel weights (bias):')
        # print(f'Custom CNN: {np.squeeze(testCNNw1k1[:-1].reshape(tfCNNw1k1.shape))} \t ({testCNNw1k1[-1]:.5f})')
        # print(f'Keras CNN: {np.squeeze(tfCNNw1k1)} \t ({np.squeeze(tfCNNw1b1):.5f})')
               
        print("Value Comparison")
        print('1st convolutional layer, 1st kernel weights (bias):')
        print(f'Custom CNN: {np.squeeze(testCNN.layers[0].weights[0][:-1]).reshape(np.squeeze(model.get_weights()[0][:,:,0,0]).shape)} \t ({np.squeeze(testCNN.layers[0].weights[0][-1]):.5f})')
        print(f'Keras CNN: {np.squeeze(model.get_weights()[0][:,:,0,0])} \t ({np.squeeze(model.get_weights()[1][0]):.5f})')
        print('Fully connected layer weights (bias):')
        print(f'Custom CNN: {np.squeeze(testCNN.layers[2].weights[0][:-1])} \t ({np.squeeze(testCNN.layers[2].weights[0][-1]):.5f})')
        print(f'Keras CNN: {np.squeeze(model.get_weights()[2])} \t ({np.squeeze(model.get_weights()[3]):.5f})')
        print(f'Output Comparison')
        print(f'Custom CNN: {np.squeeze(output_custom):.5f}')
        print(f'Keras CNN: {np.squeeze(output_tf):.5f}')

    if sys.argv[2] == 'example2':
        from parameters import generateExample2
        
        print('Example 2')
        # Call weight/data generating function
        l1k1,l1k2,l1b1,l1b2,l2k1,l2b,l3,l3b,input, output = generateExample2()
        w1k1 = np.append(l1k1.flatten(),l1b1)
        w1k2 = np.append(l1k2.flatten(),l1b2)
        w1 = np.array([w1k1,w1k2])
        w2 = np.append(l2k1.flatten(),l2b)
        w2 = np.array([w2])
        w3 = np.append(l3.flatten(),l3b)
        w3 = np.array([w3])
        
        #save initial values
        testCNNw1k1 = w1k1
        testCNNw1k2 = w1k2
        testCNNw2 = w2
        testCNNw3 = w3
        
        # Creating the custom Neural Network
        testCNN = NeuralNetwork(input_size=[7,7], loss='mse', learning_rate=learningRate, verbose=True)
        testCNN.addLayer('convolutional',2,'logistic',kernel_size=3,weights=w1)
        testCNN.addLayer('convolutional',1,'logistic',kernel_size=3,weights=w2)
        testCNN.addLayer('flatten',0,'logistic')
        testCNN.addLayer('fullyconnected',1,'logistic',weights=w3)

        #Create a feed forward network
        model=Sequential()
        # Add convolutional layers, flatten, and fully connected layer
        model.add(layers.Conv2D(2,3,input_shape=(7,7,1),activation='sigmoid')) 
        model.add(layers.Conv2D(1,3,activation='sigmoid'))
        model.add(layers.Flatten())
        model.add(layers.Dense(1,activation='sigmoid'))
        # Call weight/data generating function
        #setting weights and bias of first layer.
        l1k1=l1k1.reshape(3,3,1,1)
        l1k2=l1k2.reshape(3,3,1,1)
        w1=np.concatenate((l1k1,l1k2),axis=3)
        #setting weights and bias of second layer.
        w2=l2k1.reshape(3,3,2,1)
       
        tfCNNw1k1 = l1k1
        tfCNNw1b1 = l1b1[0]        
        tfCNNw1k2 = l1k2
        tfCNNw1b2 = l1b2[0]
        tfCNNw2k1 = l2k1
        tfCNNw2b2 = l2b
        tfCNNw3 = np.transpose(l3)
        tfCNNb3 = l3b
        
        model.layers[0].set_weights([w1,np.array([l1b1[0],l1b2[0]])]) #Shape of weight matrix is (w,h,input_channels,kernels)
        model.layers[1].set_weights([w2,l2b])
        model.layers[3].set_weights([np.transpose(l3),l3b])
        
        #Setting input. Tensor flow is expecting a 4d array since the first dimension is the batch size (here we set it to one), and third dimension is channels
        img=np.expand_dims(input,axis=(0,3))
    
        sgd = optimizers.SGD(learning_rate=learningRate)
        model.compile(loss='MSE', optimizer=sgd, metrics=['accuracy'])
        history=model.train_on_batch(img,output)
        output_tf = model.predict(img)
        
        testCNN.train([input],[output],epochs=1)      
        output_custom = testCNN.calculate(input)  
        #print needed values.
        # np.set_printoptions(precision=5)        
        # print('model output after:')
        # print(model.predict(img))
        np.set_printoptions(precision=5)
        # print("Initial Comparision")
        # print('1st convolutional layer, 1st kernel weights (bias):')
        # print(f'Custom CNN: {np.squeeze(testCNNw1k1[:-1].reshape(tfCNNw1k1.shape))} \t ({testCNNw1k1[-1]:.5f})')
        # print(f'Keras CNN: {np.squeeze(tfCNNw1k1)} \t ({np.squeeze(tfCNNw1b1):.5f})')
               
        print("Value Comparison")
        print('1st convolutional layer, 1st kernel weights (bias):')
        print(f'Custom CNN: {np.squeeze(testCNN.layers[0].weights[0][:-1]).reshape(np.squeeze(model.get_weights()[0][:,:,0,0]).shape)} \t ({np.squeeze(testCNN.layers[0].weights[0][-1]):.5f})')
        print(f'Keras CNN: {np.squeeze(model.get_weights()[0][:,:,0,0])} \t ({np.squeeze(model.get_weights()[1][0]):.5f})')
        print('1st convolutional layer, 2nd kernel weights (bias):')
        print(f'Custom CNN: {np.squeeze(testCNN.layers[0].weights[1][:-1]).reshape(np.squeeze(model.get_weights()[0][:,:,0,1]).shape)} \t ({np.squeeze(testCNN.layers[0].weights[1][-1]):.5f})')
        print(f'Keras CNN: {np.squeeze(model.get_weights()[0][:,:,0,1])} \t ({np.squeeze(model.get_weights()[1][1]):.5f})')
        print('2nd convolutional layer weights (bias):')
        print(f'Custom CNN: {np.squeeze(testCNN.layers[1].weights[0][:-1]).reshape(np.squeeze(model.get_weights()[2][:,:,:,0]).shape)} \t ({np.squeeze(testCNN.layers[1].weights[0][-1]):.5f})')
        print(f'Keras CNN: {np.squeeze(model.get_weights()[2][:,:,:,0])} \t ({np.squeeze(model.get_weights()[3]):.5f})')
        print('Fully connected layer weights (bias):')
        print(f'Custom CNN: {np.squeeze(testCNN.layers[3].weights[0][:-1])} \t ({np.squeeze(testCNN.layers[3].weights[0][-1]):.5f})')
        print(f'Keras CNN: {np.squeeze(model.get_weights()[4])} \t ({np.squeeze(model.get_weights()[5]):.5f})')
        print(f'Output Comparison')
        print(f'Custom CNN: {np.squeeze(output_custom)}')
        print(f'Keras CNN: {np.squeeze(output_tf)}')
        
        
        
    if sys.argv[2] == 'example3':
        from parameters import generateExample3
        
        print('Example 3')
        # Call weight/data generating function
        l1k1,l1k2,l1b1,l1b2,l3,l3b,input, output = generateExample3()

        w1k1 = np.append(l1k1.flatten(),l1b1)
        w1k2 = np.append(l1k2.flatten(),l1b2)
        w1 = np.array([w1k1,w1k2])
        w3 = np.append(l3.flatten(),l3b)
        w3 = np.array([w3])
        
        #save initial values
        testCNNw1k1 = w1k1
        testCNNw1k2 = w1k2
        testCNNw3 = w3
        
        # Creating the custom Neural Network
        testCNN = NeuralNetwork(input_size=[8,8], loss='mse', learning_rate=learningRate, verbose=True)
        testCNN.addLayer('convolutional',2,'logistic',kernel_size=3,weights=w1)
        testCNN.addLayer('maxpooling',1,'logistic',kernel_size=2)
        testCNN.addLayer('flatten',0,'logistic')
        testCNN.addLayer('fullyconnected',1,'logistic',weights=w3)

        #Create a feed forward network
        model=Sequential()
        # Add convolutional layers, flatten, and fully connected layer
        model.add(layers.Conv2D(2,3,input_shape=(8,8,1),activation='sigmoid')) 
        model.add(layers.MaxPool2D(pool_size=(2, 2),strides=2,padding='valid'))
        model.add(layers.Flatten())
        model.add(layers.Dense(1,activation='sigmoid'))
        # Call weight/data generating function
        #setting weights and bias of first layer.
        l1k1=l1k1.reshape(3,3,1,1)
        l1k2=l1k2.reshape(3,3,1,1)
        w1=np.concatenate((l1k1,l1k2),axis=3)
       
        tfCNNw1k1 = l1k1
        tfCNNw1b1 = l1b1[0]        
        tfCNNw1k2 = l1k2
        tfCNNw1b2 = l1b2[0]
        tfCNNw3 = np.transpose(l3)
        tfCNNb3 = l3b
        
        model.layers[0].set_weights([w1,np.array([l1b1[0],l1b2[0]])]) #Shape of weight matrix is (w,h,input_channels,kernels)
        model.layers[3].set_weights([np.transpose(l3),l3b])
        
        #Setting input. Tensor flow is expecting a 4d array since the first dimension is the batch size (here we set it to one), and third dimension is channels
        img=np.expand_dims(input,axis=(0,3))
    
        sgd = optimizers.SGD(learning_rate=learningRate)
        model.compile(loss='MSE', optimizer=sgd, metrics=['accuracy'])
        history=model.train_on_batch(img,output)
        output_tf = model.predict(img)
        
        testCNN.train([input],[output],epochs=1)
        output_custom = testCNN.calculate(input)  
        #print needed values.
        np.set_printoptions(precision=5)
        # print("Initial Comparision")
        # print('1st convolutional layer, 1st kernel weights (bias):')
        # print(f'Custom CNN: {np.squeeze(testCNNw1k1[:-1].reshape(tfCNNw1k1.shape))} \t ({testCNNw1k1[-1]:.5f})')
        # print(f'Keras CNN: {np.squeeze(tfCNNw1k1)} \t ({np.squeeze(tfCNNw1b1):.5f})')
               
        print("Value Comparison")
        print('1st convolutional layer, 1st kernel weights (bias):')
        print(f'Custom CNN: {np.squeeze(testCNN.layers[0].weights[0][:-1]).reshape(np.squeeze(model.get_weights()[0][:,:,0,0]).shape)} \t ({np.squeeze(testCNN.layers[0].weights[0][-1]):.5f})')
        print(f'Keras CNN: {np.squeeze(model.get_weights()[0][:,:,0,0])} \t ({np.squeeze(model.get_weights()[1][0]):.5f})')
        print('1st convolutional layer, 2nd kernel weights (bias):')
        print(f'Custom CNN: {np.squeeze(testCNN.layers[0].weights[1][:-1]).reshape(np.squeeze(model.get_weights()[0][:,:,0,1]).shape)} \t ({np.squeeze(testCNN.layers[0].weights[1][-1]):.5f})')
        print(f'Keras CNN: {np.squeeze(model.get_weights()[0][:,:,0,1])} \t ({np.squeeze(model.get_weights()[1][1]):.5f})')
        print('Fully connected layer weights (bias):')
        print(f'Custom CNN: {np.squeeze(testCNN.layers[3].weights[0][:-1])} \t ({np.squeeze(testCNN.layers[3].weights[0][-1]):.5f})')
        print(f'Keras CNN: {np.squeeze(model.get_weights()[2])} \t ({np.squeeze(model.get_weights()[3]):.5f})')
        print(f'Output Comparison')
        print(f'Custom CNN: {np.squeeze(output_custom):.5f}')
        print(f'Keras CNN: {np.squeeze(output_tf):.5f}')
        


    


