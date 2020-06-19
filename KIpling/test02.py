from nn_model import NN_Model,Activation_functions
import numpy
import imageio
import matplotlib.pyplot
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.7
# activation
activation_func = Activation_functions.sigmoid

# create instance of neural network
n = NN_Model(input_nodes,hidden_nodes,output_nodes, learning_rate,activation_func)

training_data_file = open("C:/Users/1/Desktop/projects/ML/mnist_dataset/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()



epochs = 100

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.Train(inputs, targets)
        pass
    pass


# test the neural network with our own images

# load image data from png files into an array
print("loading ... my_own_images/2828_my_own_image.png")
img_array = imageio.imread('C:/Users/1/Desktop/projects/ML/my_own_images/2828_my_own_image.png', as_gray=True)
    
# reshape from 28x28 to list of 784 values, invert values
img_data  = 255.0 - img_array.reshape(784)
    
# then scale data to range from 0.01 to 1.0
img_data = (img_data / 255.0 * 0.99) + 0.01
print("min = ", numpy.min(img_data))
print("max = ", numpy.max(img_data))

# plot image
matplotlib.pyplot.imshow(img_data.reshape(28,28), cmap='Greys', interpolation='None')

# query the network
outputs = n.query(img_data)
print(outputs)

# the index of the highest value corresponds to the label
label = numpy.argmax(outputs)
print("network says ", label)
