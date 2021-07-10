import numpy as np

# helper function
def tsr_list_checker(Tsr_list, children, id):
    if Tsr_list is not None:
        for c in Tsr_list:
            if id not in c.children:
                c.children[id] = 1   
            else:
                 c.children[id] += 1
    return children
    

  
class Tensor(object):
    
    def __init__(self, data, id = None, autograd = False, Tsr_list = None, operation = None):
        super().__init__()
        
        self.data     = np.array(data)
        # automatic gradient calculation
        self.autograd = autograd
        self.grad     = None
        # list of given Tesnors
        self.Tsr_list = Tsr_list
        # matrix operation, +-*
        self.operation= operation 
        # number of relatives
        self.children = {}
        
        if (id is None):
            self.id = np.random.randint(0,100000)
        else:
            self.id = id
        
        
        self.children = tsr_list_checker(self.Tsr_list,self.children,self.id)
    
    def all_children_grads_accounted_for(self):
        for id, cnt in self.children.items():
            if(cnt != 0):
                return False
        return True
        
   
       
    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data,
                          autograd  = True,
                          Tsr_list  = [self, other],
                          operation = "add")
        return Tensor(self.data + other.data)
        
    # operation substarct
    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data,
                          autograd  = True,
                          Tsr_list  = [self,other],
                          operation =  "sub")
        return Tensor(self.data - other.data)
        
    # negative operation
    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * -1,
                          autograd  = True,
                          Tsr_list  = [self],
                          operation = "neg")
        return Tensor(self.data * -1)
        
    # multiplication
    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data,
                          autograd  = True,
                          Tsr_list  = [self,other],
                          operation = "mul")
        return Tensor(self.data * other.data) 
    
    # sum       
    def sum(self, dim):
        if self.autograd:
            return Tensor(self.data.sum(dim),
                          autograd  = True,
                          Tsr_list  = [self],
                          operation = "sum_"+str(dim))
        return Tensor(self.data.sum(dim))
          
    # Matrix multiplication
    def mm(self, x):
        if(self.autograd):
            return Tensor(self.data.dot(x.data),
                          autograd  = True,
                          Tsr_list  = [self,x],
                          operation = "mm")
        return Tensor(self.data.dot(x.data))
        
    # ACTIVATION FUNCTIONS
    
    def sigmoid(self):
        if self.autograd:
            return Tensor(1 / (1 + np.exp(-self.data)),
                          autograd  = True,
                          Tsr_list  = [self],
                          operation = "sigmoid")
        return Tensor(1 / (1 + np.exp(-self.data)))
    
    def relu(self):
        if self.autograd:
            return Tensor(self.data * (self.data > 0),
                          autograd  = True,
                          Tsr_list  = [self],
                          operation = "relu")
        return Tensor(self.data * (self.data > 0))
        
    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data),
                          autograd  = True,
                          Tsr_list  = [self],
                          operation = "tanh")
        return Tensor(np.tanh(self.data))
        
    def expand(self, dim, copies):
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_data = self.data.repeat(copies).reshape(list(self.data.shape) + [copies]).transpose(trans_cmd)
        
        if self.autograd:
            return Tensor(new_data,
                          autograd  = True,
                          Tsr_list  = [self],
                          operation = "expand_"+str(dim))
        return Tensor(new_data)
    
    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(),
                          autograd  = True,
                          Tsr_list  = [self],
                          operation = "transpose")
        
        return Tensor(self.data.transpose())
    
    def index_select(self, indices):
        if self.autograd:
            new = Tensor(self.data[indices.data],
                         autograd  = True,
                         Tsr_list  = [self],
                         operation = "index_select")
            new.index_select_indices = indices
            return new
        return Tensor(self.data[indices.data])
    
    def cross_entropy(self, target_indices):
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis = len(self.data.shape)-1,
                                       keepdims = True)
        t = target_indices.data.flatten()
        p = softmax_output.reshape(len(t),-1)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p) * (target_dist)).sum(1).mean()
    
        if self.autograd:
            out = Tensor(loss,
                         autograd  = True,
                         Tsr_list  = [self],
                         operation = "cross_entropy")
            out.softmax_output = softmax_output
            out.target_dist = target_dist
            
            return out
            
    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return str(self.data.__str__())
    
    
    def backward(self, grad = None, grad_origin = None):
        if self.autograd:
            if grad is None:
                grad = Tensor(np.ones_like(self.data))
                
            if grad_origin is not None:
                if self.children[grad_origin.id] == 0:
                    raise Exception("cannot backprop more than once")
                else:
                    self.children[grad_origin.id] -= 1
                    
            if self.grad is None:
                self.grad = grad
                
            else:
                self.grad += grad
            # grads must not have grads of their own
            assert grad.autograd == False
            
            # operation checker
            if(self.Tsr_list is not None and 
               (self.all_children_grads_accounted_for() or 
                grad_origin is None)):
                
                if self.operation == "add":
                    self.Tsr_list[0].backward(self.grad)
                    self.Tsr_list[1].backward(self.grad)
                
                if self.operation == "neg":
                    self.Tsr_list[0].backward(self.grad.__neg__())
            
                if self.operation == "sub":
                    self.Tsr_list[0].backward(Tensor(self.grad.data))
                    self.Tsr_list[1].backward(Tensor(self.grad.__neg__().data))
            
                if self.operation == "mul":
                    new = self.grad * self.Tsr_list[1]
                    self.Tsr_list[0].backward(new)
                    new = self.grad * self.Tsr_list[0]
                    self.Tsr_list[1].backward(new)
             
                if self.operation == "mm":
                    c0 = self.Tsr_list[0]
                    c1 = self.Tsr_list[1]
                    new = self.grad.mm(c1.transpose())
                    c0.backward(new)
                    new = self.grad.transpose().mm(c0).transpose()
                    c1.backward(new)
            
                if self.operation == "transpose":
                     self.Tsr_list[0].backward(self.grad.transpose())
            
                if "sum" in self.operation:
                    dim = int(self.operation.split("_")[1])
                    self.Tsr_list[0].backward(self.grad.expand(dim, self.Tsr_list[0].data.shape[dim]))

                if "expand" in self.operation:
                    dim = int(self.Tsr_list.split("_")[1])
                    self.Tsr_list[0].backward(self.grad.sum(dim))
            
                if self.operation == "sigmoid":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.Tsr_list[0].backward(self.grad * (self * (ones - self)))
                
                if self.operation == "tanh":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.Tsr_list[0].backward(self.grad * (ones - (self * self)))
                    
                if self.operation == "relu":
                    ones = Tensor(np.ones_like(self.grad.data))
                    self.Tsr_list[0].backward(self.grad * ones)
                
                if self.operation == "index_select":
                    new_grad = np.zeros_like(self.Tsr_list[0].data)
                    indices_ = self.index_select_indices.data.flatten()
                    grad_ = grad.data.reshape(len(indices_), -1)
                    for i in range(len(indices_)):
                        new_grad[indices_[i]] += grad_[i]
                    self.Tsr_list[0].backward(Tensor(new_grad))
                    
                if self.operation == "cross_entropy":
                    dx = self.softmax_output - self.target_dist
                    self.Tsr_list[0].backward(Tensor(dx))
                  
    
        
a = Tensor([1,2,3,4,5], autograd=True)
b = Tensor([2,2,2,2,2], autograd=True)
c = Tensor([5,4,3,2,1], autograd=True)
d = a + b
e = b + c
f = d + e
print(f.backward(Tensor(np.array([1,1,1,1,1]))))
print(f)
print(b.grad.data == np.array([2,2,2,2,2]))
