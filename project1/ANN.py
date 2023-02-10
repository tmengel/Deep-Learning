############################################
# Authors: Tanner Mengel and Ian Cox
# Date: 2/4/2022
# Project 1 - Artificial Neural Networks

############################################

import numpy as np
import sys

class Neuron:
    
    def __init__(self, input_dim, activation, learning_rate=0.1, weights=None, verbose=False):
        self.input_dim = input_dim
        self.activation = activation
        if activation not in ['linear', 'logistic']:
            raise ValueError(f'Activation function {activation} not supported')
        self.learning_rate = learning_rate
        if weights is None: self.weights = np.random.rand(input_dim)
        else: self.weights = weights
        if self.weights.shape[0] != input_dim:
            raise ValueError(f'Input dim does not match the dim of weights, input dim: {input_dim}, weights dim: {self.weights.shape[0]}')
        self.output = None
        self.inputs = []
        self.dW = []
        
        if verbose:
            print('Initialized Neuron with the following architecture:')
            print(f'Input Dim: {input_dim}, Activation Function: {activation}')
            print(f'Learning Rate: {learning_rate}, Weights: {weights}')
            print(f'Weights Shape: {weights.shape}')
    
    def active(self, input):
        if self.activation == 'linear': return input
        elif self.activation == 'logistic': return 1/(1+np.exp(-input))
        else: raise ValueError(f'Activation function {self.activation} not supported')
        
    def calculate(self, X):
        self.inputs = X
        input = np.dot(self.weights, X)
        output = self.active(input)
        self.output = output
        return output
    
    #This method returns the derivative of the activation function with respect to the net   
    def activationderivative(self):
        # print('activationderivative')  
        if self.activation == "linear": return 1
        elif self.activation == "logistic": return self.output*(1-self.output)
        else : raise ValueError("Activation function not found")
      
    #This method calculates the partial derivative for each weight and returns the delta*w to be used in the previous layer
    def calcpartialderivative(self, wtimesdelta):
        # print('calcpartialderivative') 
        delta = wtimesdelta*self.activationderivative()
        self.dW = self.inputs*delta
        return delta*self.weights
               
    #Simply update the weights using the partial derivatives and the leranring weight
    def updateweight(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate*self.dW[i]
    
      
class Layer:
    
    def __init__(self, neuron_num, activation, input_num, learning_rate=0.1, weights=None, verbose=False):
        self.neuron_num = neuron_num
        self.activation = activation
        self.learning_rate = learning_rate
        self.input_num = input_num
        if weights is None:
            weights = []
            for i in range(neuron_num):
                weights.append(np.random.rand(input_num))
        self.weights = weights
        self.neurons = []
        for i in range(neuron_num):
            weight = weights[i]
            inputdim = weight.shape[0]
            if inputdim != input_num:
                raise ValueError(f'Input dim does not match the dim of weights, input dim: {inputdim}, weights dim: {input_num}')
            self.neurons.append(Neuron(input_dim=inputdim, activation=activation, learning_rate=learning_rate, weights=weight))
        self.outputs = []
        if verbose:
            print('Initialized Layer with the following architecture:')
            print(f'Neuron Number: {neuron_num}, Activation Function: {activation}')
            print(f'Input Number: {input_num}, Learning Rate: {learning_rate}')
            print(f'Weights: {weights}')
            print(f'Weights Shape: {weights.shape}')

    def calculate(self, X):
        input = X
        output = []
        for neuron in self.neurons:
            output.append(neuron.calculate(input))
        return output
    
    def calcwdeltas(self, dloss):
        #given the next layer's w*delta, should run through the neurons calling calcpartialderivative() for each (with the correct value), sum up its own w*delta, and then update the weights (using the updateweight() method). I should return the sum of w*delta.          
        deltaw = np.zeros(self.input_num)
        for i in range(self.neuron_num):
            deltaw += self.neurons[i].calcpartialderivative(dloss[i])
            self.neurons[i].updateweight()    
        #print('deltaw', deltaw)
        return deltaw
        
            
class NeuralNetwork:
    
    def __init__(self, num_of_layers, num_of_neurons, input_size, activation, loss, output_size=1, learning_rate=0.1, weights=None,verbose=False):
        self.num_of_layers = num_of_layers
        self.num_of_neurons = num_of_neurons
        if type(num_of_neurons) != list:
            raise ValueError(f'Number of neurons must be a list, not {type(num_of_neurons)}')
        if len(num_of_neurons) != num_of_layers:
            raise ValueError(f'Number of neurons does not match the number of layers, neurons: {len(num_of_neurons)}, layers: {num_of_layers}')
        self.input_size = input_size
        self.activation = activation
        if type(activation) != list:
            print('Constant activation function for all layers')
            self.activation = []
            for i in range(num_of_layers):
                self.activation.append(activation)
            self.activation.append(activation)
        if len(self.activation) != num_of_layers + 1:
            if len(self.activation) == num_of_layers:
                print("Output layer activation function not specified, using previous activation function")
                self.activation.append(activation[-1])
            else:
                print("Activation function not specified for all layers, using previous activation function")
                for i in range(num_of_layers - len(self.activation)):
                    self.activation.append(activation[-1])
            
        self.loss = loss
        if loss not in ['mse', 'bce']:
            raise ValueError(f'Loss function {loss} not supported')
        
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.architecture = [input_size] + num_of_neurons + [output_size]
        if weights is None:
            weights = []
            for i in range(len(self.architecture) - 1):
                weights.append(np.random.rand(self.architecture[i+1], self.architecture[i]+1))  
        self.weights = weights           
        self.layers = []
        for i in range(num_of_layers):
            weight = weights[i]
            neurons = weight.shape[0]
            inputdim = weight.shape[1]
            if neurons != num_of_neurons[i]:
                raise ValueError(f'Number of neurons in layer {i} does not match the dim of weights, neurons: {neurons}, weights dim: {num_of_neurons[i]}')
            self.layers.append(Layer(neuron_num=neurons, activation=activation[i], input_num=inputdim, learning_rate=learning_rate, weights=weight, verbose=verbose)) 
        self.layers.append(Layer(neuron_num=weights[-1].shape[0], activation=activation[-1], input_num=weights[-1].shape[1], learning_rate=learning_rate, weights=weights[-1], verbose=verbose))
        self.outputs = []
        
        if verbose:
            print('Initialized Neural Network with the following architecture:')
            print(f'Input Size: {input_size}, Output Size: {output_size}, Hidden Layers: {num_of_layers}')
            print(f'Neurons per Hidden Layer: {num_of_neurons}')
            print(f'Activation Function: {activation}, Loss Function: {loss}')
            print(f'Learning Rate: {learning_rate}')
            print(f'Weights: {weights}')
            for weight in self.weights:
                print(weight.shape)
            
    def calculate(self,X):
        input = X
        for layer in self.layers:
            input = np.append(input, 1)
            output = layer.calculate(input)
            #print(output)
            input = output
        return output
    
    def calculateloss(self,yhat,y):
        if self.loss == 'mse': 
            loss = 0
            for i in range(len(yhat)):
                loss += (yhat[i] - y[i])**2
            loss = loss/len(yhat)
            return loss
        elif self.loss == 'bce':
            loss = 0
            for i in range(len(yhat)):
                loss += -y[i]*np.log(yhat[i]) - (1-y[i])*np.log(1-yhat[i])
            loss = loss/len(yhat)
            return loss
        else : raise ValueError(f'Loss function {self.loss} not supported')
        
    def lossderiv(self,yhat,y):
        if self.loss == 'mse':
            dloss = 0
            for i in range(len(yhat)):
                dloss = 2*(yhat[i] - y[i])
            dloss = dloss/len(yhat)
            return dloss
        elif self.loss == 'bce':
            dloss = 0
            for i in range(len(yhat)):
                dloss = -y[i]/yhat[i] + (1-y[i])/(1-yhat[i])
            dloss = dloss/len(yhat)
            return dloss
        else : raise ValueError(f'Loss function {self.loss} not supported')
    
    def train(self, X, Y, epochs=1, verbose=False):
        # check if X is multiple samples
        # if type(X[0]) != list:
        #     X = [X] 
        #     Y = [Y]
        for epoch in range(epochs):
            for i in range(len(X)):
                #print(f'X: {X[i]} Y: {Y[i]}')
                yhat = self.calculate(X[i])
                #print(f'Yhat: {yhat}')
                dloss = self.lossderiv(yhat, Y[i])
                #print(f'Dloss: {dloss}')
                #check if dloss is a list
                dloss = [dloss]
               # print(f'Dloss: {dloss}')
                for j in range(len(self.layers)-1, -1, -1):
                    dloss = self.layers[j].calcwdeltas(dloss)   
            yhat = []
            ytest = []
            for i in range(len(X)):
                yhat.append(*self.calculate(X[i]))
                ytest.append(*Y[i])
            loss = self.calculateloss(yhat, Y)
            if epoch % 10000 == 0:
                print(f'Epoch: {epoch}, Loss: {loss}')
        
         
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
        example = NeuralNetwork(1,[2],len(x),["logistic","logistic"],"mse",len(y),learningRate,w)
        example.train(x,y,1,True)
        print(example.weights)
        
    elif(sys.argv[2]=='and'):
        test = NeuralNetwork(num_of_layers=1, num_of_neurons=[1], input_size=2, activation=['linear', 'logistic'], loss='bce', output_size=1, learning_rate=learningRate)
        x = np.array([[0,0],[0,1],[1,0],[1,1]])
        y = np.array([[0],[0],[0],[1]])
        print('training network')
        test.train(x,y,epochs=10000,verbose=True)
        yhat = []
        for i in range(len(x)):
            yhat.append(*test.calculate(x[i]))
        print("And")
        for i in range(len(x)):
            print(f'X: {x[i]}, Y: {y[i]}, Yhat: {yhat[i]}')
        print('Total Loss: ', test.calculateloss(yhat, y))
        
    elif(sys.argv[2]=='xor'):
        test = NeuralNetwork(num_of_layers=1, num_of_neurons=[15], input_size=2, activation=['logistic', 'logistic'], loss='bce', output_size=1, learning_rate=0.1)
        x = np.array([[0,0],[1,0],[0,1],[1,1]])
        y = np.array([[0],[1],[1],[0]])
        print('training network')
        test.train(x,y,epochs=50000,verbose=True)
        yhat = []
        for i in range(len(x)):
            yhat.append(*test.calculate(x[i]))
        print("XOR")
        for i in range(len(x)):
            print(f'X: {x[i]}, Y: {y[i]}, Yhat: {yhat[i]}')
        print('Total Loss: ', test.calculateloss(yhat, y))

        print("Training single perceptron")
        test = NeuralNetwork(num_of_layers=0, num_of_neurons=[], input_size=2, activation=['logistic'], loss='bce', output_size=1, learning_rate=0.3)
        test.train(x,y,epochs=100000,verbose=True)
        yhat = []
        for i in range(len(x)):
            yhat.append(*test.calculate(x[i]))
        print("XOR Single Perceptron")
        for i in range(len(x)):
            print(f'X: {x[i]}, Y: {y[i]}, Yhat: {yhat[i]}')
        print('Total Loss: ', test.calculateloss(yhat, y))