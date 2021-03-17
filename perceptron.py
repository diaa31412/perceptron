# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 00:53:09 2021

@author: Diyaa
"""

import numpy as np
import pandas as pd
import csv

data = pd.read_csv('input1.csv', sep=',',header=None)

class Perceptron(object):

    def __init__(self, no_of_inputs, threshold=100, learning_rate=0.01):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)
           
    def predict(self, inputs):
        for row in range(0,inputs.shape[0]):
                x1 = inputs[0]
                x2 = inputs[1]
                
                summation =x1 * self.weights[0] + x2 * self.weights[1]
                
   
            #print("X1 is: ",x1 * self.weights[0], "X2 is :",x2 * self.weights[1] ,"Y :",summation)
        if summation > 0:
          activation = 1
        else:
          activation = -1 
        
        return activation

    def train(self, training_inputs, labels):
        out = np.array([],ndmin=3)
        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                print("prediction is ", prediction , "  The actual value is ", label)
                self.weights[0] += self.learning_rate * (label - prediction)
                self.weights[1] += self.learning_rate * (label - prediction)
                
                output=np.append(out,[self.weights[0],self.weights[1],prediction])
        return (output)
            
                


data = pd.read_csv('input1.csv', sep=',',header=None)
dataset = np.array(data)
x =dataset[:,0:2]
y=dataset[...,2]
re = Perceptron(len(data[0]))
re.train(x,y)
