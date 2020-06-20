from nn_model import NN_Model,Activation_functions
import numpy


x_train = numpy.array([[1,0,1],
               [0,0,1],
               [1,1,0],
               [0,1,0],])

y_train = numpy.array([[1,
                0,
                1,
                0,]]).T
input_nodes     = 3
hidden_nodes    = 3
output_nodes    = 3
learning_rate   = 0.3
activation_func = Activation_functions.sigmoid
example = example = NN_Model(input_nodes,hidden_nodes,output_nodes,learning_rate,activation_func)
epochs  = 1000

for e in range(epochs):
    example.Train(x_train,y_train)
    pass
pass

test = [[1,0,1]]

outputs = example.query(test)
print(outputs[0][0])
