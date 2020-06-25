# Importing LIbraries
from matplotlib import pyplot
from math import cos, sin, atan
# Neuron class 

class Neuron():
    def __init__(self, x, y):
        self.x  = x
        self.y  = y
    # Draw cirlces 
    def Draw(self, neuron_R):
        self.circle  =  pyplot.Circle((self.x,self.y), radius = neuron_R,fill = False)
        pyplot.gca().add_patch(self.circle)
# Layers
class Layer():
    def __init__(self, network, amount_of_neurons, amount_of_neurons_in_widest_layer):
        
        self.vertical_distance_between_layers    = 15
        self.horizontal_distance_between_neurons = 10
        self.neuron_R = 1
        self.amount_of_neurons_in_widest_layer = amount_of_neurons_in_widest_layer
        self.previous_layer = self.get_previous_layer(network)
        self.y       = self.layer_calculation_y()
        self.neurons = self.Neurons_init(amount_of_neurons)
        
    def Neurons_init(self, amount_of_neurons):
        neurons = []
        self.x  =  self.layer_calculation_x(amount_of_neurons)
        
        for iterations in range(amount_of_neurons):
            self.neuron = Neuron(self.x,self.y)
            neurons.append(self.neuron)
            self.x += self.horizontal_distance_between_neurons
        return neurons
        
    def layer_calculation_x(self,amount_of_neurons):
        return self.horizontal_distance_between_neurons * (self.amount_of_neurons_in_widest_layer - amount_of_neurons) / 2
        
    def layer_calculation_y(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0
            
    def get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None
            
    def line_between_two_neurons(self, neuron1, neuron2):
        self.angle        = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        self.x_adjustment = self.neuron_R * sin(self.angle)
        self.y_adjustment = self.neuron_R * cos(self.angle)
        self.line         = pyplot.Line2D((neuron1.x - self.x_adjustment, neuron2.x + self.x_adjustment), (neuron1.y - self.y_adjustment, neuron2.y + self.y_adjustment))
        pyplot.gca().add_line(self.line)
        
    def Draw(self, layerType=0):
        for neuron in self.neurons:
            neuron.Draw( self.neuron_R)
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.line_between_two_neurons(neuron, previous_layer_neuron)
        
        self.x_text = self.amount_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        
        if layerType == 0:
            pyplot.text(self.x_text, self.y, 'Input Layer', fontsize = 10)
        elif layerType == -1:
            pyplot.text(self.x_text, self.y, 'Output Layer', fontsize = 10)
        else:
            pyplot.text(self.x_text, self.y, 'Hidden Layer '+str(layerType), fontsize = 10)

class NN():
    def __init__(self,number_of_neurons_in_widest_layer):
        
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layertype = 0
    def Add_layer(self, number_of_neurons):
        self.layer = Layer(self, number_of_neurons,self.number_of_neurons_in_widest_layer)
        self.layers.append(self.layer)
    
    def draw(self):
        pyplot.figure()
        for i in range(len(self.layers)):
            self.layer = self.layers[i]
            if i == len(self.layers)-1:
                i = -1
            self.layer.Draw( i )
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title( 'Neural Network architecture', fontsize=15 )
        pyplot.show()


class DrawNN():
    def __init__( self, neural_network ):
        self.neural_network = neural_network
        print(self.neural_network,' nn')

    def draw( self ):
        widest_layer = max( self.neural_network )
        network = NN( widest_layer )
        for l in self.neural_network:
            network.Add_layer(l)
        network.draw()
        

def main(object):
    object = object
    network = DrawNN(object)
    network.draw()
          
