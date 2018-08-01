# multi layered neural network 
# neural network do not have 'bias' so input cannot contain 'zero'


import numpy as np
import scipy.special


class NeuralNetwork :
    
    # initialise neural network
    # 'inode' number of nodes in input layer / number of input
    # 'hnode' is a list containing number of nodes in each hidden layer
    # 'onode' number of nodes in output layer / number of output
    # (10, [4,5,6,3], 4, .5)
    def __init__(self, inode, hnode, onode, learning_rate) :
        
        self.inode = inode
        self.onode = onode
        self.hnode = hnode
        self.hnode_count = len(hnode)
        self.lr = learning_rate
        
        self.weight = []
        
        # weight between input and first hidden layer
        w_temp = np.random.normal(0.0, pow(self.inode, -0.5), (self.hnode[0], self.inode))
        (self.weight).append(w_temp)
        
        
        # weight between all hidden layer
        i = 0
        for i in range(1,self.hnode_count) :
            w_temp = np.random.normal(0.0, pow(self.hnode[i-1], -0.5), (self.hnode[i], self.hnode[i-1]))
            (self.weight).append(w_temp)
            pass
        
        
        # weight between last hidden and output layer
        w_temp = np.random.normal(0.0, pow(self.hnode[i], -0.5), (self.onode, self.hnode[i]))
        (self.weight).append(w_temp)
        
        self.weight_len = len(self.weight)
       
        pass
    
    def activation(self,inputs) :
        
        return scipy.special.expit(inputs)
    
    
    # 'train' method is used to train neural network
    # 'input_list' 1D numpy array or 1D list containing inputs
    # 'output_list' 1D numpy array or 1D list containing output/result
    def train(self, input_list, output_list) :
        
        inputs = (np.array(input_list, ndmin = 2).T)
        expected = (np.array(output_list, ndmin = 2).T)
        
        
        # output by each layer is stored in outputs
        outputs = []  
        for i in range(len(self.weight)) :
            inputs = np.dot(self.weight[i], inputs)
            inputs = self.activation(inputs)
            outputs.append(inputs)
            pass
        len_outputs = len(outputs)  # self.hnode_count - 1
        
        # finding error of each layer using backpropogation and storing in errors
        errors = []
        errors.append(expected - outputs[len_outputs-1])
        j = self.weight_len - 1 
        for i in range(1,len_outputs) :
            temp = np.dot((self.weight[j]).T, errors[i-1])
            errors.append(temp)
            j = j - 1
            pass
        
        # updating weight using gradient desent algo
        j = len(errors) - 1
        self.weight[0] += self.lr * np.dot((errors[j] * outputs[0] * (1.0 - outputs[0])),  np.array(input_list, ndmin = 2))
        j = j - 1
        for i in range(1, self.weight_len) :
            self.weight[i] += self.lr * np.dot((errors[j] * outputs[i] * (1.0 - outputs[i])), outputs[i-1].T)
            j = j - 1
            pass
        pass
    
    # 'query' method is used to test neural network
    # 'input_list' 1D numpy array or 1D list containing inputs
    # output of this method is 2D numpy array of size (n x 1) 
    def query(self, input_list) :
        
        inputs = (np.array(input_list, ndmin = 2)).T
        
        for i in range(len(self.weight)) :
            inputs = np.dot(self.weight[i], inputs)
            inputs = self.activation(inputs)
            pass
        
        return inputs

        
    
    pass

  
