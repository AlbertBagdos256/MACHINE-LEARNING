#----------------------------#
# Author: Albert Bagdasarov
# Date: 09.06.20
#----------------------------#

__version__ = '1.0'


# Importing Libraries
import numpy
# library for plotting arrays
import matplotlib.pyplot

class Activation_functions:
    def sigmoid(x):
        return 1 / (1 + numpy.exp(-x))
    def ReLU(x):
        return x * (x > 0)
    

class NN_Model:
    # initialise the neural network
    def __init__(self,inputnodes,hidenodes,outputnodes,learningrate,activation):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hidenodes
        self.onodes = outputnodes
        # Link weight matrices, hidden_weight,output_weight
        self.hidden_weight  = numpy.random.normal(0.0, pow(self.inodes, -0.5),(self.hnodes, self.inodes))
        self.output_weight  = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        # Learning rate
        self.lr = learningrate
        # activation function
        self.activation_function = activation   
        

    # training process     
    def Train(self,inputs_list,targets_list):
        # convert inputs list to 2d array
        self.inputs  = numpy.array(inputs_list, ndmin=2).T
        self.targets = numpy.array(targets_list, ndmin=2).T
        # calcualte signals into hidden_layer
        self.hidden_inputs  = numpy.dot(self.hidden_weight,self.inputs)
        # calculate the signals emerging from hidden layer
        self.hidden_outputs = self.activation_function(self.hidden_inputs)
        # calculate signals into final output layer
        self.final_inputs   = numpy.dot(self.output_weight, self.hidden_outputs)
        # calculate the signals emerging from final output layer
        self.final_outputs  = self.activation_function(self.final_inputs)

        # output layer error is the (taregt - actual)
        self.output_errors  = self.targets - self.final_inputs
        # hidden layer error is the output_errors, split by weights,, recombined at hidden nodes
        self.hidden_errors  = numpy.dot(self.hidden_weight.T,self.output_errors)
         # update the weights for the links between the hidden and output layers
        self.output_weight += self.lr * numpy.dot((self.output_errors * self.final_outputs * (1.0 - self.final_outputs)), numpy.transpose(self.hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.hidden_weight += self.lr * numpy.dot((self.hidden_errors * self.hidden_outputs * (1.0 - self.hidden_outputs)), numpy.transpose(self.inputs))
        

        
    
    #interrogation of a neural network
    def query(self,inputs_list):
        # convert inputs list to 2d array
        self.inputs         = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        self.hidden_inputs  = numpy.dot(self.hidden_weight, self.inputs)
        # calculate the signals emerging from hidden layer
        self.hidden_outputs = self.activation_function(self.hidden_inputs)
        
        # calculate signals into final output layer
        self.final_inputs   = numpy.dot(self.output_weight, self.hidden_outputs)
        # calculate the signals emerging from final output layer
        self.final_outputs  = self.activation_function(self.final_inputs)
        return self.final_outputs
        
    


'''
Example:

model =NN_Model(NN_Model(input_nodes,hidden_nodes,output_nodes,learning_rate,activation_func))

'''
