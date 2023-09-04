#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf


# In[2]:


from tensorflow import keras


# In[3]:


import pandas as pd


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df = pd.read_csv("E:\\age.csv")


# In[8]:


df.head()


# In[13]:


df.shape


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age', 'afforablity']], df.have_insurance, test_size = 0.2, random_state = 25)


# In[15]:


X_train


# In[20]:


len(X_train)


# In[21]:


X_train_scaled = X_train.copy()


# In[22]:


X_train_scaled['age'] = X_train_scaled['age']/100
X_test_scaled = X_test.copy()
X_test_scaled['age'] = X_test_scaled['age']/100


# In[23]:


X_train_scaled


# In[26]:


model = keras.Sequential([
    keras.layers.Dense(1, input_shape = (2,), activation = 'sigmoid', kernel_initializer = 'ones', bias_initializer = 'zeros')
])

model.compile(
    optimizer = 'adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

model.fit(X_train_scaled, y_train, epochs = 5000)


# In[27]:


model.evaluate(X_test_scaled, y_test)


# In[28]:


X_test_scaled


# In[29]:


model.predict(X_test_scaled)


# In[30]:


y_test


# In[31]:


coef, intercept = model.get_weights()


# In[32]:


coef, intercept


# In[34]:


def sigmoid(x):
    import math
    return 1/1 + math.exp(-x)
sigmoid(18)


# In[35]:


def prediction_function(age, afforability):
    weighted_sum = coef[0]*afforability + intercept
    return sigmoid(weighted_sum)


# In[36]:


prediction_function(.47, 1)


# In[37]:


prediction_function(.18, 1)


# In[39]:


def log_loss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_new = [max(i, epsilon) for i in y_predicted]
    y_predicted_new = [min(i, epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new) + (1-y_true)* np.log(1-y_predicted_new))
    


# In[63]:


def sigmoid_numpy(X):
    return 1/1 + np.exp(-X)
sigmoid_numpy(np.array([12, 0, 1]))


# In[72]:


def gradient_descent(age, afforablity, y_true, epochs):
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
        
        print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
    return w1, w2, bias


# In[73]:


gradient_descent(X_train_scaled['age'], X_train_scaled['afforablity'], y_train, 1000)


# In[74]:


model.evaluate(X_test_scaled, y_test)


# In[75]:


X_test_scaled


# In[76]:


model.predict(X_test_scaled)


# In[77]:


y_test


# In[78]:


coef, intercept = model.get_weights()
coef, intercept


# In[79]:


def sigmoid(x):
    import math
    return 1/(1 + math.exp(-x))
sigmoid(18)


# In[80]:


def prediction_function(age, afforablity):
    weighted_sum = coef[0]*age+coef[1]*afforablity + intercept
    return sigmoid(weighted_sum)


# In[81]:


prediction_function(.47, 1)


# In[83]:


prediction_function(.18, 1)


# In[ ]:




