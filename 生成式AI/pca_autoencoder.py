import numpy as np

import cv2

import matplotlib.pyplot as plt

dataX = np.load('mnist.npy').reshape([-1, 28*28])

cov = np.cov(dataX.T)

val, vec = np.linalg.eig(cov)

sort = np.argsort(-np.abs(val))

dim = 64

w1 = np.abs(vec[:, 0:dim])

z1 = np.dot(dataX, w1)

w2 = np.random.rand(*w1.T.shape)/dim

lr = 0.001
Epoch = 30
batch = 20

for i in range(Epoch):
    
    for j in range(len(dataX)//batch):
        
        bx = z1[j*batch:(j+1)*batch]
        
        z2 = np.dot(bx, w2)
        
        delta2 = (z2-dataX[j*batch:(j+1)*batch])
        
        grad2 = np.dot(bx.T,delta2)/len(bx)
        
        w2 -= lr*grad2
        
    z2 = np.dot(z1, w2)
    delta2 = (z2-dataX)
    print(str(i)+' : '+str(np.mean(delta2)))
    
A = z1[1]
B = z1[10]

dx = np.arange(0, 1, 0.01)

V_AB = B-A

vecs = A + V_AB.reshape(1,-1)*dx.reshape([-1,1])

imgs = np.dot(vecs, w2)
top = np.max(imgs,1).reshape([-1,1])
bot = np.min(imgs,1).reshape([-1,1])
imgsN = (imgs-bot)/(top-bot)*255

table = imgsN.reshape([10,10,28,28]).transpose([0,2,1,3]).reshape([28*10, 28*10])
plt.imshow(table)