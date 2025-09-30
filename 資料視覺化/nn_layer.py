import numpy as np

class NN():
    def __init__(self,inputNode, NodeNum = 128, Activation = 'relu'):
        self.weight = np.random.randn(inputNode, NodeNum)/inputNode
        self.activation = Activation
        self.grad = 0
        self.input = 0
        self.Z = 0
        self.A = 0
        
    def forward(self, input):
        """
        Parameters
        ----------
        input : shape(Batch,Dim)

        """
        self.input = input
        self.Z = np.dot(input, self.weight)
        
        if self.activation=='relu':
            self.A = self.Z.copy()
            self.A[self.A<0] = 0
            return self.A
        
        if self.activation=='sigmoid':
            # A = Z.copy()
            self.A = 1/(1+np.exp(-self.Z))
            return self.A
        
        if self.activation=='none':
            self.A = self.Z
            return self.A
        
    def backward(self, delta):
        """
        Parameters
        ----------
        delta : shape(Batch,Dim)

        Returns
        -------
        Next delta vectors
        """
        if self.activation=='none':
            self.grad = np.dot(self.input.T, delta)/len(self.input)
            return np.dot(delta, self.weight.T)
        
        if self.activation=='sigmoid':
            self.grad = np.dot(self.input.T, delta*(self.A)*(1-self.A))/len(self.input)
            return np.dot(delta*(self.A)*(1-self.A), self.weight.T)
        
        if self.activation=='relu':
            self.grad = np.dot(self.input.T, delta*(self.Z>0))/len(self.input)
            return np.dot(delta*(self.Z>0), self.weight.T)
        
    def backwardFinal(self, label):
        
        delta = (self.A-label)
        
        if self.activation=='none': # MSE loss
            self.grad = np.dot(self.input.T, delta)/len(self.input)
            return np.dot(delta, self.weight.T)
        
        if self.activation=='sigmoid': # Cross Entropy loss
            self.grad = np.dot(self.input.T, delta)/len(self.input)
            return np.dot(delta, self.weight.T)
        
        if self.activation=='relu': # MSE loss
            self.grad = np.dot(self.input.T, delta*(self.Z>0))/len(self.input)
            return np.dot(delta*(self.Z>0), self.weight.T)
        
    def update(self, lr = 1e-4):
        self.weight = self.weight-lr*self.grad
        
    def save(self, Layer = 'L1'):
        np.save('./'+ Layer+ 'weight.npy',self.weight)
        np.save('./'+ Layer+ 'activate.npy',self.activation)
        
    def load(self, Layer = 'L1'):
        self.weight = np.load('./'+ Layer+ 'weight.npy', allow_pickle=True)
        self.activation = np.load('./'+ Layer+ 'activate.npy', allow_pickle=True)