import numpy as np
import pandas as pd
import scipy.stats
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy import genfromtxt
#Values to generate data
np.random.seed(12)
mean1 = [1, 0]
cov1 = [[1, 0.75], [0.75, 1]]
x1= np.random.multivariate_normal(mean1, cov1, 1000)
mean2 = [0, 1.5]
cov2 = [[1, 0.75], [0.75, 1]]  
x2 = np.random.multivariate_normal(mean2, cov2, 1000)
train_data=np.vstack((x1,x2)).astype(np.float32)
train_label = np.hstack(((np.ones(1000)),(np.zeros(1000))))
xt1 = np.random.multivariate_normal(mean1, cov1, 500)
xt2= np.random.multivariate_normal(mean2, cov2, 500)
test_data=np.vstack((xt1,xt2)).astype(np.float32)
test_label = np.hstack(((np.ones(500)),(np.zeros(500))))


#Xa=X.astype(int)

 #whole training data

# Y=["Label0","Label1"]# includes the 2 lables, 0 and 1
# #np.savetxt("foo2.csv", Xa, delimiter=",")
# # my_data = genfromtxt('foo.csv', delimiter=',')
# Xa[:,2]=1

# print(my_data)



#testing data
# test_set0 = np.append(np.random.multivariate_normal(mu1,sigma1,500),np.zeros((500,1)),axis=1)
# test_set1 = np.append(np.random.multivariate_normal(mu2,sigma2,500),np.ones((500,1)),axis=1)
# X_test=test_set0+test_set1 # whole testing data
# Y_test=["Label0","Label1"] # includes the 2 labels, 0 and 1

# print("Train data is",Xa)
# print("Test data is",X_test)

# #logistic regression 
# #sigmoid function(activation function)
# def Sigmoid(z):
# 	return 1/(1+np.exp(-z))

# #cross entropy(objective function)
# def cross_entropy(X,y):
#     """
#     X is the output from fully connected layer (num_examples x num_classes)
#     y is labels (num_examples x 1)
#     	Note that y is not one-hot encoded vector. 
#     	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
#     """
#     m = y.shape[0]
#     p = softmax(X)
#     # We use multidimensional array indexing to extract 
#     # softmax probability of the correct label for each sample.
#     # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
#     log_likelihood = -np.log(p[range(m),y])
#     loss = np.sum(log_likelihood) / m
#     return loss


# # #stopping condition
# # if l1_norm<0.001 or count==100000:
# # 	break

class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=2000):
        self.lr = lr
        self.num_iter = num_iter

    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
         return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            print(self.__loss(h,y))
            print(self.theta)


        
    
    def predict_prob(self, X):
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X)


model=LogisticRegression(lr=0.01,num_iter=3000)
model.fit(train_data,train_label)
predict=model.predict(test_data).round()
print((predict==test_label).mean()*100)





