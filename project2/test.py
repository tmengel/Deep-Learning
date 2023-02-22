import numpy as np
import sys

from project2 import *

learningRate = 0.1
test = NeuralNetwork(2,'bce',learningRate)
#test.addLayer('maxpooling',1,'logistic',2)
test.addLayer('fullyconnected',2,'logistic')
#x = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
#x = np.random.rand(4,4)
x = [[0,1]]
print(x[0])
print('training network')
#print(test.architecture)

res = test.calculate(x)
print(res)
