#!/usr/bin/env python
# coding: utf-8

# In[119]:


import numpy as np
import tensorflow as tf


# In[120]:


from tensorflow import keras


# In[121]:


import pandas as pd


# In[122]:


import matplotlib.pyplot as plt


# In[123]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[124]:


df = pd.read_csv("E:\\age.csv")


# In[125]:


df.head()


# In[126]:


df.shape


# In[127]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age', 'afforablity']], df.have_insurance, test_size = 0.2, random_state = 25)


# In[128]:


X_train


# In[129]:


len(X_train)


# In[130]:


X_train_scaled = X_train.copy()


# In[131]:


X_train_scaled['age'] = X_train_scaled['age']/100
X_test_scaled = X_test.copy()
X_test_scaled['age'] = X_test_scaled['age']/100


# In[132]:


X_train_scaled


# In[133]:


model = keras.Sequential([
    keras.layers.Dense(1, input_shape = (2,), activation = 'sigmoid', kernel_initializer = 'ones', bias_initializer = 'zeros')
])

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

model.fit(X_train_scaled, y_train, epochs = 5000)


# In[134]:


model.evaluate(X_test_scaled, y_test)


# In[135]:


X_test_scaled


# In[136]:


model.predict(X_test_scaled)


# In[137]:


y_test


# In[138]:


coef, intercept = model.get_weights()


# In[139]:


coef, intercept


# In[140]:


def sigmoid(x):
    import math
    return 1/1 + math.exp(-x)
sigmoid(18)


# In[141]:


def prediction_function(age, afforability):
    weighted_sum = coef[0]*afforability + intercept
    return sigmoid(weighted_sum)


# In[142]:


prediction_function(.47, 1)


# In[143]:


prediction_function(.18, 1)


# In[144]:


def log_loss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_new = [max(i, epsilon) for i in y_predicted]
    y_predicted_new = [min(i, epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new) + (1-y_true)* np.log(1-y_predicted_new))
    


# In[145]:


def sigmoid_numpy(X):
    return 1/1 + np.exp(-X)
sigmoid_numpy(np.array([12, 0, 1]))


# In[159]:


class myNN:
    def __init__(self):
        self.w1 = 1
        self.w2 = 1
        self.bias = 0
        
    
    def fit(self, X, y, epochs, loss_thresold):
        self.w1, self.w2, self.bias = self.gradient_descent(X['age'], X['afforablity'], y, epochs, loss_thresold)
    def predcit(self, X_test):
        weighted_sum = self.w1 * X_test['age'] + self.w2 * X_test['afforablity'] + self.bias
        return sigmoid_numpy(weighted_sum)
    
    def gradient_descent(self, age, afforablity, y_true, epochs, loss_thresold):
        #w1, w1, bias
        w1 = w2 = 1
        bias = 0
        rate = 0.5
        n = len(age)

        for i in range(epochs):
            weighted_sum = w1 * age + w2 * afforablity + bias
            y_predicted = sigmoid_numpy(weighted_sum)

            loss = log_loss(y_true, y_predicted)

            w1d = (1/n)*np.dot(np.transpose(age),(y_predicted-y_true))
            w2d = (1/n)*np.dot(np.transpose(afforablity),(y_predicted-y_true))

            bias_d = np.mean(y_predicted-y_true)


            w1 = w1 - rate*w1d
            w2 = w2 - rate*w2d
            bias = bias - rate * bias_d
            
            if i%50 == 0:
                print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
            
            if loss<=loss_thresold:
                print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
                break
        return w1, w2, bias


# In[160]:


customModel = myNN()
customModel.fit(X_train_scaled, y_train, epochs = 500, loss_thresold = 0.4631)


# In[167]:


coef, intercept


# customModel.predict(X_test_scaled)

# In[168]:


model.predict(X_test_scaled)


# In[ ]:




