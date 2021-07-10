from Tensor import Tensor
import numpy as np



class Layer(object):
    
    def __init__(self):
        self.parameters = list()
        
    def get_parameters(self):
        return self.parameters     


class Tanh(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.tanh()
    
    
class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        return input.sigmoid()


class SGD(object):
    def __init__(self, parameters, alpha=0.1):
        self.parameters = parameters
        self.alpha = alpha
    
    def zero(self):
        for p in self.parameters:
            p.grad.data *= 0
        
    def step(self, zero=True):
        for p in self.parameters:
            p.data -= p.grad.data * self.alpha
            if(zero):
                p.grad.data *= 0

class Linear(Layer):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0/(n_inputs))
        self.weight = Tensor(W, autograd=True)
        self.bias = Tensor(np.zeros(n_outputs), autograd=True)
        
        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, input):
        return input.mm(self.weight)+self.bias.expand(0,len(input.data))


class Sequential(Layer):
    def __init__(self, layers = list()):
        super().__init__()  
        self.layers = layers 
        
    def add(self, layer):
        self.layers.append(layer)    
        
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
        
    def get_parameters(self):
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params
        
        
class CrossEntropyLoss(object):
    def __init__(self):
        super().__init__()
    def forward(self, input, target):
        return input.cross_entropy(target)
        
        
class MSELoss(Layer):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        return ((pred - target)*(pred - target)).sum(0)

class Embedding(Layer):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        # this random initialiation style is just a convention from word2vec
        self.weight = Tensor((np.random.rand(vocab_size, dim) - 0.5) / dim, autograd=True)
        self.parameters.append(self.weight)
    def forward(self, input):
        return self.weight.index_select(input)



# Automatically hyper-parameters optimization
class Tuner:
    def __init__(self):
        super().__init__()
        self.weight          = []
        self.learning_rate   = []
        self.alpha           = []
        self.epochs          = 0
        self.input_amount    = 0
    def get_parameters(self):
        # getting weights configuration
        for i in range(self.input_amount):
            self.random_weights = np.random(0,1)
            self.weight += self.random_weights
        self.learning_rate = np.random(0,1)
        self.alpha         = np.random(0,1)
        self.epochs        = random.randint(10,100)
    def Hit(self):
        pass
        