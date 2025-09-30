import numpy as np
import matplotlib.pyplot as plt

dataX=np.load('mnist.npy')
dataY=np.load('mnistLabel.npy')

####### 僅能使用:

#np.reshape
#np.transpose
#np.repeat
#np.tile
#np.mean
#plt.imshow
#---------------


####### 將第0張圖片 ( 數字5 ) 和第1張圖片 ( 數字 0 )串在一起得到數字50的圖片 :
dataX_50 = dataX[0:2]
dataX_50 = dataX_50.transpose(1,0,2,3).reshape([28,28*2])
plt.imshow(dataX_50)


#------------------------------------------------------


####### 將第0張圖片 ( 數字5 )  放大5倍

dataX_0 = dataX[0].reshape([28,28])
dataX_0 = dataX_0.repeat(5,0).repeat(5,1)
plt.imshow(dataX_0)

#------------------------------------------------------



####### 將第0張圖片 ( 數字5 )  縮小2倍

dataX_0 = dataX[0].reshape([28,28])
dataX_0 = dataX_0.reshape([14,2,14,2])
dataX_0 = dataX_0.transpose(0,2,1,3)
dataX_0 = np.mean(dataX_0,(2,3))
plt.imshow(dataX_0)

img = np.mean(dataX[0].reshape([14,2,14,2]),(1,3))
plt.imshow(img)

#------------------------------------------------------


####### 將第資料中圖片內容為0-9數字挑出來，將之合併成2*5的大圖片
num = [1,3,5,7,2,0,13,15,17,4]
dataX_09 = []

for i in range(10):
    dataX_09.append(dataX[num[i]])
    
dataX_09 = np.array(dataX_09)
# a = dataX_09[0].reshape([28,28])
# plt.imshow(a)

dataX_09 = np.reshape(dataX_09, [2,5,28,28])
dataX_09 = np.transpose(dataX_09,[0,2,1,3])
dataX_09 = np.reshape(dataX_09, [2*28,5*28])
plt.imshow(dataX_09)
#------------------------------------------------------



####### 接續上題->將圖片從長方形(56*140)改成正方形大小(28*28)
dataX_sq = dataX_09.reshape([28,2,28,5])
dataX_sq = np.mean(dataX_sq,(1,3))
plt.imshow(dataX_sq)


#------------------------------------------------------

# 30*30
dataX_30times = dataX_sq.repeat(30,0).repeat(30,1)
dataX_30times = np.mean(dataX_30times.reshape([30,28,30,28]),(1,3))
plt.imshow(dataX_30times)
