import numpy as np
from Tensor import Tensor
from utils import SGD, Sequential, MSELoss, Tanh, Linear, Sigmoid

np.random.seed(0)

data = Tensor(np.array([[0,0],[0,1],[1,0],[1,1]]), autograd = True)
target = Tensor(np.array([[0],[1],[0],[1]]), autograd = True)

model = Sequential([Linear(2, 3), Tanh(), Linear(3, 1), Sigmoid()])
criterion = MSELoss()

optim = SGD(parameters = model.get_parameters(), alpha = 1)

for i in range(10):
    
    # Predict
    pred = model.forward(data)
    
    # Compare
    loss = criterion.forward(pred, target)
    
    # Learn
    loss.backward(Tensor(np.ones_like(loss.data)))
    optim.step()
    print(loss)