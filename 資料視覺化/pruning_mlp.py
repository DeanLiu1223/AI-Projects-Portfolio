import numpy as np
import nn_layer as ai


class network():
    
    def __init__(self):
        self.L1 = ai.NN(32*32*3, 128, Activation="none")
        self.L2 = ai.NN(128, 128)
        self.L3 = ai.NN(128, 128, Activation="none")
        self.L4 = ai.NN(128, 128)
        self.L5 = ai.NN(128, 128, Activation="none")
        self.L6 = ai.NN(128, 10, Activation="sigmoid")
    
    
    def forward(self, inputs, returnLayers=False):
        A1 = self.L1.forward(inputs)
        A2 = self.L2.forward(A1)
        A3 = self.L3.forward(A2)
        A3_2 = A1 + A3
        
        A4 = self.L4.forward(A3_2)
        A5 = self.L5.forward(A4)
        A5_2 = A3_2 + A5
        
        A6 = self.L6.forward(A5_2)
        
        if returnLayers:
            return [A1, A2, A3, A4, A5, A6]
        else:
            return A6
    
    
    def backward(self, label):
        d6 = self.L6.backwardFinal(label)
        d5 = self.L5.backward(d6)
        d4 = self.L4.backward(d5)
        
        d3 = self.L3.backward(d6 + d4)
        d2 = self.L2.backward(d3)
        d1 = self.L1.backward(d6 + d4 + d2)
        
        return d1
        
    
    def update(self, lr=0.0001):
        self.L1.update(lr)
        self.L2.update(lr)
        self.L3.update(lr)
        self.L4.update(lr)
        self.L5.update(lr)
        self.L6.update(lr)
    
    
    def fit(self, inputs, labels, Epoch=10, batch=20, lr=0.0001, verbose=1):
        
        for i in range(Epoch):
            
            if verbose==1:
                y = self.forward(inputs)
                ACC = np.sum(np.argmax(y, 1) == np.argwhere(labels == 1)[:, 1]) / len(inputs)
                print(str(i) + " : " + str(ACC))
                
            for j in range(len(inputs) // batch):
                bx = inputs[j*batch:(j+1)*batch]
                by = labels[j*batch:(j+1)*batch]
                
                self.forward(bx)
                self.backward(by)
                self.update(lr)
    
    def pruning(self):
        
        ######################### 1-Norm
        
        # w2 = self.L2.weight
        # w3 = self.L3.weight
        # w4 = self.L4.weight
        # w5 = self.L5.weight
        
        # w2Norm = np.sum(np.abs(w2), 0)
        # w4Norm = np.sum(np.abs(w4), 0)
        
        # w2pick = w2Norm > np.mean(w2Norm) + np.std(w2Norm)
        # w4pick = w4Norm > np.mean(w4Norm) + np.std(w4Norm)
        
        
        # w2New = w2[:, w2pick]
        # w3New = w3[w2pick, :]
        
        # w4New = w4[:, w4pick]
        # w5New = w5[w4pick, :]
        
        # self.L2.weight = w2New
        # self.L3.weight = w3New
        # self.L4.weight = w4New
        # self.L5.weight = w5New
        
        #-----------------------
        
        
        ######################### 2-Norm
        
        w2 = self.L2.weight
        w3 = self.L3.weight
        w4 = self.L4.weight
        w5 = self.L5.weight
        
        w2Norm = np.sqrt(np.sum(w2**2, 0))
        w4Norm = np.sqrt(np.sum(w4**2, 0))
        
        w2pick = w2Norm > np.mean(w2Norm) + np.std(w2Norm)
        w4pick = w4Norm > np.mean(w4Norm) + np.std(w4Norm)
        
        
        w2New = w2[:, w2pick]
        w3New = w3[w2pick, :]
        
        w4New = w4[:, w4pick]
        w5New = w5[w4pick, :]
        
        self.L2.weight = w2New
        self.L3.weight = w3New
        self.L4.weight = w4New
        self.L5.weight = w5New
        
        #-----------------------
        
                

dataX = np.load("cifar10.npy").reshape([-1, 32*32*3])
dataY = np.load("cifar10Label.npy").reshape([-1])
D = np.identity(10)[dataY]
        



model = network()
model.fit(dataX, D, 10, 20, 0.0001)
y = model.forward(dataX)

model.pruning()

print(np.shape(model.L1.weight))
print(np.shape(model.L2.weight))
print(np.shape(model.L3.weight))
print(np.shape(model.L4.weight))
print(np.shape(model.L5.weight))
print(np.shape(model.L6.weight))

model.fit(dataX, D, 10, 20, 0.0001)
y = model.forward(dataX)

















