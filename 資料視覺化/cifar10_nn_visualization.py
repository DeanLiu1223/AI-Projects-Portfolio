import numpy as np
import matplotlib.pyplot as plt

ans=np.load('ans.npy',allow_pickle=True)
dataX=np.load('cifar10.npy').reshape([-1,32*32*3])
dataY=np.load('cifar10Label.npy').reshape([-1])

name=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

##[飛機、汽車、鳥、貓、鹿、狗、青蛙、馬、船、卡車]

#



####### 將資料庫的資料可視化 #########

####### 計算共變異矩陣
cov = np.cov(dataX.T)

print('cov : Checked status is  '+ str(np.prod(ans[0].astype(np.float32)==cov.astype(np.float32))==1))
#---------------------------------

####### 計算特徵值與特徵向量
val, vec = np.linalg.eig(cov)

print('Eigen : Checked status is  '+ str(np.prod(ans[1].astype(np.float32)==val.astype(np.float32))==1))
#---------------------------------

####### 選擇特徵值最大的2個向量進行投影

select = np.argsort(-np.abs(val))[0:2]
dim2 = np.dot(dataX,vec[:,select])

print('dim2 : Checked status is  '+ str(np.prod(ans[2].astype(np.float32)==dim2.astype(np.float32))==1))
#---------------------------------


####### 將上面主成分數據分布圖畫出來(要標上legend)
plt.figure()

for i in range(len(name)):
    pick = dataY == i
    plt.scatter(dim2[pick,0], dim2[pick,1], label= str(name[i]), marker="x")

plt.title("Data Visualization")
plt.legend()

#---------------------------------





D=np.identity(10)[dataY]
w1=np.random.uniform(-1,1,[3072,512])/3072
w2=np.random.uniform(-1,1,[512,10])/512

EP=100

lr=0.0001
batch=50
for ep in range( EP):

    for j in range(len(dataX)//batch):  
        
    ##################################### Get Batch Data
        bx=dataX[j*batch:(j+1)*batch]
        by=D[j*batch:(j+1)*batch]  
    #-------------bx (100,784) by (100,10)---------------------------------------- End code

    ##################################### Forward Pass Z1=???

        Z1=np.dot( bx,w1)

   #----------------Z1 (100,128)------------------------------------- End code 
    ##################################### Forward Pass A1=relu(Z1)

        A1=Z1.copy()
        A1[A1<0]=0
   
   #----------------Z1 (100,128)------------------------------------- End code        
    ######################################## Forward Pass Z2=???      

        Z2=np.dot( A1 , w2)        
     #----------------Z2 (100,10)----------------------------- End code

    ######################################## Forward Pass A2=???     

        A2=1/(1+np.exp(-Z2))

   #---------------A2 (100,10)---------------------------------- End code
  
        ######################################## Backward delta2=???
    
        grad2=np.dot(A1.T,-(by-A2)) / len(bx)
        
   #-------------grad2 (128,10)------------------------------- End code
            
  ######################################## Backward delta2=???

        grad1= np.dot( bx.T,np.dot( -(by-A2),w2.T)*(Z1>0)  ) / len(bx)

   #------------grad1 (784,128)--------------------------- End code
        
        choose=np.argmax(Z2,1)
        print(str(ep)+" ACC (a batch): "+str(np.sum(choose==dataY[j*batch:(j+1)*batch])/len(by)))
        
        
        w1=w1-lr*(grad1)
        w2=w2-lr*(grad2)




      
Z1=np.dot( dataX,w1)  
A1=Z1.copy()
A1[A1<0]=0
Z2=np.dot( A1 , w2)
choose=np.argmax(Z2,1)



print('----------------------------------')
print(" TotalACC:\t"+str(np.sum(choose==dataY)/len(D)))
print('----------------------------------')



 ####### 將 Z1 的資料可視化 #########
plt.figure()

cov_Z1 = np.cov(Z1.T)
val_Z1, vec_Z1 = np.linalg.eig(cov_Z1)
s_Z1 = np.argsort(-np.abs(val_Z1))[0:2]
dim2_Z1 = np.dot(Z1,vec_Z1[:,s_Z1])

for i in range(len(name)):
    pick = dataY==i
    plt.scatter(dim2_Z1[pick,0], dim2_Z1[pick,1], label = str(name[i]),marker="x")

plt.title("Z1 Visualization")
plt.legend()
plt.show()

#---------------------------------


####### 將 Z2 的資料可視化 #########
plt.figure()

cov_Z2 = np.cov(Z2.T)
val_Z2, vec_Z2 = np.linalg.eig(cov_Z2)
s_Z2 = np.argsort(-np.abs(val_Z2))[0:2]
dim2_Z2 = np.dot(Z2,vec_Z2[:,s_Z2])

for i in range(len(name)):
    pick = dataY==i
    plt.scatter(dim2_Z2[pick,0], dim2_Z2[pick,1], label = str(name[i]),marker="x")

plt.title("Z1 Visualization")
plt.legend()
plt.show()

#---------------------------------

### w1 直方圖

"""
for i in range(w1.shape[1]):
    ep_w1 = w1[:,i]
    plt.hist(ep_w1, bins=100, label=name[i], histtype="step")
    plt.legend()
plt.title("w1 histogram")
plt.show()
"""

### w2 直方圖

for i in range(w2.shape[1]):
    ep_w2 = w2[:,i]
    plt.hist(ep_w2, bins=100, label=name[i], histtype="step")
    plt.legend()
plt.title("w2 histogram")
plt.show()


### w1 熱度圖

"""
f_w1 = w1.reshape([2*128, 2*128, 2, 5])
f_w1 = f_w1.transpose([2,0,3,1])
f_w1 = f_w1.reshape([2*2*128, 5*2*128])
plt.title("w1 heat map")
plt.imshow(f_w1)
"""

selected = w1.T[:10]  # (10, 3072)

plt.figure(figsize=(15, 4))
for i in range(10):
    img = selected[i].reshape(32, 32, 3)  # reshape 成彩色圖像
    img = (img - img.min()) / (img.max() - img.min())  # 正規化到 0~1
    plt.subplot(1, 10, i+1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Neuron {i}')
plt.suptitle("w1 weights visualized as CIFAR images")
plt.show()

### w2 熱度圖

f_w2 = w2.reshape([16, 32, 2, 5])
f_w2 = f_w2.transpose([2,0,3,1])
f_w2 = f_w2.reshape([2*16, 5*32])
plt.title("w2 heat map")
plt.imshow(f_w2)


