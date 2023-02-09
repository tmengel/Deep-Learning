############################################
# Authors: Tanner Mengel and Ian Cox
# Date: 2/4/2022
# Project 1 - Artificial Neural Networks

############################################

import numpy as np
import sys



############################################

class Activation:
    ''' 
    Class for activation functions and their derivatives also a method to create custom activation functions
    '''
    # constructor that checks if f and df are callable
    def __init__(self, f, df, name=None):
        # confirm that f and df are callable
        if not callable(f) or not callable(df):
            raise ValueError("Activation function and derivative must be callable\nExample : f = lambda x: x**2, df = lambda x: 2*x")
        self._function = f
        self._derivative = df
        self._name = name if name is not None else "custom"

    # Predefined activation functions and their derivatives
    @staticmethod
    def linear(): # static method to return a linear activation function
        return Activation(lambda x: x, lambda x: 1, "linear")
    
    @staticmethod 
    def logistic(): # static method to return a logistic activation function
        return Activation(lambda x: 1/(1+np.exp(-x)), lambda x: x * (1 - x), "logistic")
    
    # Method to return user defined activation function
    @staticmethod
    def custom(f, df, name=None): # static method to return a custom activation function
        return Activation(f, df, name if name is not None else "custom")

    @classmethod
    def get(cls, name): # class method to return an activation function from a string
        if name == "linear": return cls.linear()
        elif name == "logistic": return cls.logistic()
        else: raise ValueError("Activation function not found")
        
    # call properties to return the activation function and its derivative or the name of the activation function
    def df(self, x): return self._derivative(x)
    def f(self, x): return self._function(x)
    def name(self): return self._name
   
class Loss:
    '''
    Class for loss functions and user defined loss function 
    '''
    def __init__(self, f, df, name=None): 
        if not callable(f) or not callable(df): # confirm that f and df are callable
            raise ValueError("Loss function and derivative must be callable\nExample : f = lambda y, yhat: np.mean((y - yhat)**2), df = lambda y, yhat: 2*(yhat - y)")
        self._name = name if name is not None else "custom"
        self._function = f
        self._derivative = df
        
    # Predefined loss functions are mean squared error and binary cross entropy
    @staticmethod
    def mse(): # static method to return a mean squared error loss function
        return Loss(lambda y, yhat: np.mean(0.5*(yhat- y )**2), lambda y, yhat: -1*(yhat - y), "mse")
    @staticmethod
    def bce(): # static method to return a binary cross entropy loss function
        return Loss(lambda y, yhat: -np.mean(y*np.log(yhat) + (1-y)*np.log(1-yhat)), lambda y, yhat: (yhat - y)/(yhat*(1-yhat)), "bce")
    @staticmethod
    def custom(f, df, name=None): # static method to return a custom loss function
        return Loss(f, df, name if name is not None else "custom")
        
    @classmethod
    def get(cls, name): # class method to return a loss function from a string
        if name == "mse" or name == "MSE" or name == "mean_squared_error": return cls.mse()
        elif name == "bce" or name == "BCE" or name == "binary_cross_entropy": return cls.bce()
        else : raise ValueError("Loss function not found")
    
    # call properties to return the loss function and its derivative or the name of the loss function
    def loss(self, y, yhat): return self._function(y, yhat)  
    def derivatderivativeive(self, y, yhat): return self._derivative(y, yhat)
    def name(self): return self._name  
    
    # function to calculate the loss and derivative of the loss without creating a Loss object
    def MSE(y, yhat): return np.mean((yhat - y)**2)
    def BCE(y, yhat): return -np.mean(y*np.log(yhat) + (1-y)*np.log(1-yhat))
      
############################################

class Neuron:
    '''
    This class contains the methods for a single neuron
    '''
    # Neuronal properties and setters
    @property
    def weights(self): return self._weights
    @property
    def inputs(self): return self._inputs
    @property
    def outputs(self): return self._outputs
    @property
    def nets(self): return self._nets
    @property
    def delta(self): return self._delta
    @property
    def dW(self): return self._dW
    @property
    def input_dim(self): return self._input_dim
    
    @weights.setter
    def weights(self, x): self._weights = x
    @inputs.setter
    def inputs(self, x): self._inputs = x
    @outputs.setter
    def outputs(self, x): self._outputs = x
    @nets.setter
    def nets(self, x): self._nets = x
    @delta.setter
    def delta(self, x): self._delta = x
    @dW.setter
    def dW(self, x): self._dW = x
    # @input_dim.setter
    # def input_dim(self, x): self._input_dim = x
        
    def __init__(self, input_dim, activation, learning_rate=0.1, weights=None): 
        self.learning_rate = learning_rate
        self.weights = weights if weights is not None else np.random.rand(input_dim+1) # if weights is not specified, initialize the weights randomly.. added weight to act as the bias
        # check that the weights are the correct shape
        if self.weights.shape != (input_dim+1,): 
            raise ValueError("Weights must be a vector of length input_dim", self.weights.shape,input_dim+1)
        
        self._input_dim = input_dim
        # check if activation a string or an Activation object
        self.activation = activation if isinstance(activation, Activation) else Activation.get(activation)
        # print('neuron constructor')
    
             
    #This method returns the activation of the net
    def activate(self,net):
        # print('activate')
        return self.activation.f(net)
    
    #Calculate the output of the neuron should save the input and output for back-propagation.   
    def calculate(self,input):
        # print('calculate')
        self.inputs = input
        self.nets = np.dot(self.weights, self.inputs)
        self.outputs = self.activation.f(self.nets)
        return self.outputs
        
    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):
        # print('activationderivative')  
        return self.activation.df(self.outputs)
      
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        # print('calcpartialderivative') 
        self.delta = wtimesdelta*self.activationderivative()
        self.dW = self.inputs*self.delta
        return self.delta*self.weights
               
    #Simply update the weights using the partial derivatives and the leranring weight
    def updateweight(self):
        # print('updateweight') 
        self.weights -= float(self.learning_rate)*self.dW
     
        
############################################
       
class FullyConnected:
    '''
    Class manages neurons in a fully connected layer, initializes the neurons and calculates the output of the layer
    '''
    
    # Layer properties and setters
    @property
    def weights(self): return self._weights
    @property
    def inputs(self): return self._inputs
    @property
    def outputs(self): return self._outputs
    @property
    def deltaw(self): return self._deltaw

    @weights.setter
    def weights(self, x): self._weights = x
    @inputs.setter
    def inputs(self, x): self._inputs = x
    @outputs.setter
    def outputs(self, x): self._outputs = x
    @deltaw.setter
    def deltaw(self, x): self._deltaw = x
 
    
    def __init__(self, neuron_num, activation, input_num, learning_rate=0.1, weights=None):
        # print('fully connected constructor') 
        self.weights = weights if weights is not None else np.random.rand(neuron_num,input_num+1) #if weights is not specified, initialize the weights randomly.
        self.neurons = [Neuron(input_num, activation, learning_rate, self.weights[i,:]) for i in range(neuron_num)] #create a list of neurons with the correct number of inputs and weights
                
    #calcualte the output of all the neurons in the layer and return a vector with those values (go through the neurons and call the calcualte() method)      
    def calculate(self, input):
        # print('fully connected calculate') 
        input = np.append(input,1)
        self.inputs = input 
        self.outputs = np.array([neuron.calculate(input) for neuron in self.neurons])        
        return self.outputs
    
    #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its own w*delta, and then update the weights (using the updateweight() method). I should return the sum of w*delta.          
    def calcwdeltas(self, dloss):
        # print('fully connected calcwdeltas') 
        # go through the neurons and call calcpartialderivative() for each
        self.deltaw  = np.zeros(self.weights.shape)
        # print(dloss[i],self.neurons[i].calcpartialderivative(dloss[i]),self.weights[i])
        for i in range(len(self.neurons)):
            neuron = self.neurons[i]
            self.deltaw[i] += np.float64(neuron.calcpartialderivative(dloss[i]))
            neuron.updateweight()
        # update the weights
        return self.deltaw

############################################

   
class NeuralNetwork:
    ''''
    Class manages the layers of the neural network, initializes the layers and calculates the output of the network
    '''
    # Network properties and setters
    @property
    def weights(self): return self._weights
    @property
    def inputs(self): return self._inputs
    @property
    def outputs(self): return self._outputs
    @property
    def inputsize(self): return self._inputsize
    @property
    def network(self): return self._network
    @property
    def loss(self): return self._loss
    @property
    def dloss(self): return self._dloss
    @property
    def lossResult(self): return self._lossResult
    
    
    @weights.setter
    def weights(self, x): self._weights = x
    @inputs.setter
    def inputs(self, x): self._inputs = x
    @outputs.setter
    def outputs(self, x): self._outputs = x
    @network.setter
    def network(self, x): self._network = x
    @inputsize.setter
    def inputsize(self, x): self._inputsize = x
    @loss.setter
    def loss(self, x): self._loss = x
    @dloss.setter
    def dloss(self, x): self._dloss = x
    @lossResult.setter
    def lossResult(self, x): self._lossResult = x
    
    
    def __init__(self, num_of_layers, num_of_neurons, input_size, activation, loss, output_size=1, learning_rate=0.1, weights=None):
        # print('neural network constructor')
        self._activations = activation
        self._learning_rate = learning_rate
        self._inputsize = input_size
        # check that the number of neurons is the correct length
        if len(num_of_neurons) != num_of_layers:
            print(num_of_neurons,num_of_layers)
            raise ValueError("num_of_neurons must be a vector of length num_of_layers")

        self.architecture = [input_size , *num_of_neurons , output_size] # add the input size and output size to the architecture
        print(self.architecture)
        # check that the weights are the correct shape
        # for i in range(len(self.architecture)-1):
        #     if weights[i].shape != (self.architecture[i],self.architecture[i+1]):
        #         raise ValueError("Weights must be a vector of length num_of_layers")
            
        # Setting random weights based on the architecture of the network
        # extra weight added to account for the bias
        if weights is not None:
            self.weights = weights
        else:
            weights = [np.random.rand(num_of_neurons[0],input_size+1)]
            for i in range(len(num_of_neurons)-1):
                weights.append(np.random.rand(num_of_neurons[i+1],num_of_neurons[i]+1))
            weights.append(np.random.rand(output_size,num_of_neurons[-1]+1))
            self.weights = weights

        self.network = [FullyConnected(self.architecture[i+1],activation[i],self.architecture[i],learning_rate,self.weights[i]) for i in range(len(self.architecture)-1)]
        # Get the loss function from the Loss class
        self.loss = Loss.get(loss)
        
    #Given an input, calculate the output (using the layers calculate() method)
    def calculate(self,input):
        # set the input
        self.inputs = input
        # go through the layers and calculate the output, setting the output of one layer to the input of the next
        # final output should be stored in self.outputs
        for layer in self.network:
            self.outputs = layer.calculate(input)
            input = self.outputs
        # return the output
        return self.outputs
        
    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)
    def calculateloss(self,yp,y):
        # print('calculate loss')
        return self.loss.loss(yp,y)
    
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)        
    def lossderiv(self,yp,y):
        # print('lossderiv')
        return self.loss.derivatderivativeive(yp,y)
        
    def doEpoch(self,x,y):
        # feed forward
        self.outputs = self.calculate(x)
        # calculate the derivative of the loss
        self.dloss = self.lossderiv(self.outputs,y)
        calcLoss = self.calculateloss(self.outputs,y)
        self.lossResult.append(calcLoss)
        # print(calcLoss)
        
        # backpropagate starting with last layer
        for i in reversed(range(len(self.network))):
            if i==0:
                continue
            layer = self.network[i]
            deltaw = layer.calcwdeltas(self.dloss)
        return(calcLoss)
    
    #Given a single input and desired output preform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values         
    def train(self,x,y,numiter=1):
        self.lossResult = []
        # if numiter==0:
        #     while self.dloss>0.1:
        #         if x.ndim==1:
        #             self.doEpoch(x,y)
        # else:
        for iter in range(numiter):
            if x.ndim==1:
                self.doEpoch(x,y)
            else:
                ll = 0
                for it in range(len(x)):
                    ll += self.doEpoch(x[it],y[it])/len(x)
                print(ll)
        
        
        # print('train')
            
            
if __name__=="__main__":
    learningRate = sys.argv[1]
    print('Using a learning rate of',learningRate)
    if (len(sys.argv)<3):
        print('a good place to test different parts of your code')
    
    elif (sys.argv[2]=='example'):
        print('run example from class (single step)')
        w=np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])
        x =np.array([0.05,0.1])
        y = np.array([0.01,0.99])
        example = NeuralNetwork(1,[2],len(x),["logistic","logistic"],"mse",len(y),learningRate)
        example.train(x,y,10)
        print(example.weights)
        
    elif(sys.argv[2]=='and'):
        xs = np.array([[0,0],[1,0],[0,1],[1,1]])
        ys = np.array([0,0,0,1])
        andnn = NeuralNetwork(2,[2,1],len(xs[0]),["logistic","linear","logistic"],"mse",1,learningRate)
        andnn.train(xs,ys,100)
        print('learn and')
        
    elif(sys.argv[2]=='xor'):
        xs = np.array([[0,0],[1,0],[0,1],[1,1]])
        ys = np.array([0,1,1,0])
        print('learn xor')
