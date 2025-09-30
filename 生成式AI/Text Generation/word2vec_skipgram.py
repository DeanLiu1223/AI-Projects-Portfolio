import pandas as pd
import GAI_v2 as gai
import numpy as np

reader = pd.read_csv('Gossiping-QA-Dataset-2_0.csv')[0:10000]

reader = reader.dropna()
reader = reader.drop_duplicates()
reader = reader.to_numpy()

specialTokens = ['/B', '/S', '/E']

stream = ''.join(specialTokens[0] + reader[:, 0] + specialTokens[1] + reader[:, 1] + specialTokens[2])

t = gai.tokenizer()
t.train(stream)
dataset = t.split(stream)
tokens = t.tokenize(dataset)

X = []
Y = []
l = []

for i in range(len(tokens) - 5 + 1):
    
    if '/B' in dataset[i + 1:i + 3]:
        continue
    
    if '/E' in dataset[i + 1:i + 3]:
        continue
    
    # Add Positive Label
    
    X.append(tokens[i + 2])
    X.append(tokens[i + 2])
    X.append(tokens[i + 2])
    X.append(tokens[i + 2])
    
    Y.append(tokens[i + 0])
    Y.append(tokens[i + 1])
    Y.append(tokens[i + 3])
    Y.append(tokens[i + 4])
    
    l.append(1)
    l.append(1)
    l.append(1)
    l.append(1)
    
    # Add Negative Label
    
    X.append(tokens[i + 2])
    X.append(tokens[i + 2])
    X.append(tokens[i + 2])
    X.append(tokens[i + 2])
    
    for i in range(100):
        pick = np.random.randint(0, len(tokens))
        
        if dataset[pick] not in dataset[i:i + 4]:
            Y.append(tokens[pick])
        
        if len(Y) == len(X):
            break
    
    l.append(0)
    l.append(0)
    l.append(0)
    l.append(0)
    
Data = np.concatenate([X, Y, l]).reshape([3, -1]).T

dim = 128

size = len(t.getDict())
w1 = np.random.randn(size, dim) / size
w2 = np.random.randn(dim, size) / dim

EP = 100
lr = 0.05
batch = 50

for i in range(EP):
    
    for j in range(len(Data) // batch):
        bx = Data[j*batch : (j + 1)*batch, 0]
        by = Data[j*batch : (j + 1)*batch, 1]
        bl = Data[j*batch : (j + 1)*batch, 2]
        
        # Forward Pass
        
        z1 = w1[bx]
        z2 = np.sum(z1 * w2[:, by].T, 1)
        
        A2 = 1 / (1 + np.exp(-z2))
        
        loss = -np.mean(bl * np.log(A2) + (1 - bl) * np.log(1 - A2))
        
        print('Epoch : '+ str(i) + ', Loss : ' + str(loss))
        
        # Backward
        
        delta = (A2 - bl)
        
        grad2 = z1*delta.reshape([-1, 1])
        grad1 = w2[:, by].T*delta.reshape([-1, 1])
        
        for k in range(len(bx)):
            w1[bx[k], :] = w1[bx[k], :] - lr*grad1[k]
            w2[:, by[k]] = w2[:, by[k]] - lr*grad2[k]