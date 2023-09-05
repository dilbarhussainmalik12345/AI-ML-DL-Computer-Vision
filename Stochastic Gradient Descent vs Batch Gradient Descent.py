#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from matplotlib import pyplot as plt


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv("E:\\houseprice.csv")


# In[5]:


df


# In[6]:


df.drop('price', axis = 'columns')


# In[7]:


from sklearn import preprocessing

sx = preprocessing.MinMaxScaler()
sy = preprocessing.MinMaxScaler()


# In[8]:


scaled_X = sx.fit_transform(df.drop('price', axis = 'columns'))


# In[9]:


scaled_X


# In[10]:


scaled_y = sy.fit_transform(df['price'].values.reshape(df.shape[0], 1))


# In[11]:


scaled_y


# In[15]:


w = np.ones(shape = [2])


# In[20]:


def batch_gradient_descent(X, y_true, epochs, learning_rate = 0.01):
    number_of_features = X.shape[1]
    w = np.ones(shape = (number_of_features))
    b = 0
    total_samples = X.shape[0]
    
    cost_list = []
    epoch_list = []
    
    for i in range(epochs):
        y_predicted = np.dot(w, scaled_X.T) + b #w1 * area + w2*bedrooms
        
        w_grad = -(2/total_samples)*(X.T.dot(y_true - y_predicted))
        w_grad = -(2/total_samples)*np.sum(y_true - y_predicted)
        
        
        w = w-learning_rate * w_grad
        b = b-learning_rate * w_grad
        
        cost = np.mean(np.square(y_true-y_predicted))
        
        if i%10 == 0:
            cost_list.append(cost)
            epoch_list.append(i)
    return w, b, cost, cost_list, epoch_list   
w,b,cost, cost_list, epoch_list = batch_gradient_descent(scaled_X, scaled_y.reshape(scaled_y.shape[0], ), 500)
w,b,cost
        


# In[21]:


plt.xlabel("epoch")
plt.ylabel("cost")
plt.plot(epoch_list, cost_list)


# In[22]:


sx.transform([[2600, 3]])


# In[23]:


w


# In[28]:


sy.inverse_transform([[1,0.5,0]])


# In[29]:


def predict(area, bedrooms, w, b):
    scaled_X = sx.transform([[area, bedrooms]])[0]
    
    scaled_price = w[0]*scaled_X[0] + w[1] * scaled_X[1] + b
    return sy.inverse_transform([[scaled_price]])
    pass

predict(2600,3,w,b)


# In[31]:


predict(3600, 4, w,b)


# In[32]:


import random
random.randint(0,6)


# In[39]:


def stochastic_gradient_descent(X, y_true, epochs, learning_rate = 0.01):
    number_of_features = X.shape[1]
    
    w = np.ones(shape=(number_of_features))
    b = 0
    total_samples = X.shape[0]
    
    cost_list = []
    epoch_list = []
    
    for i in range(epochs):
        random_index = random.randint(0,total_samples-1)
        sample_x = X[random_index]
        sample_y = y_true[random_index]
        y_predicted = np.dot(w, sample_x.T) + b
        
        w_grad = -(2/total_samples)*(X.T.dot(y_true - y_predicted))
        w_grad = -(2/total_samples)*np.sum(y_true - y_predicted)
        
        
        w = w-learning_rate * w_grad
        b = b-learning_rate * w_grad
        
        cost = np.square(y_true-y_predicted)
        
        if i%100== 0:
            cost_list.append(cost)
            epoch_list.append(i)
    return w, b, cost, cost_list, epoch_list   

w_sgd, b_sgd, cost_sgd, cost_list_sgd, epoch_list_sgd = stochastic_gradient_descent(scaled_X, scaled_y.reshape(scaled_y.shape[0],),10000)
w_sgd, b_sgd, cost_sgd


# In[40]:


w,b,cost


# In[43]:


plt.xlabel("epoch")
plt.ylabel("cost")
plt.plot(epoch_list_sgd, cost_list_sgd)
plt.show()


# In[44]:


predict(2600, 3 , w_sgd, b_sgd)


# In[ ]:




