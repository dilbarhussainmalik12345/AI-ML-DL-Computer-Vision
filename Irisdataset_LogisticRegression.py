#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import pandas as pd
import numpy as np


# In[6]:


os.chdir("C:\\python")
os.getcwd()


# In[7]:


dataset = pd.read_csv("iris.csv")
dataset


# In[8]:


x=dataset.iloc[:,:4].values
x


# In[9]:


y=dataset.iloc[:,4].values
y


# In[10]:


from sklearn.preprocessing import LabelEncoder


# In[11]:


lbl = LabelEncoder()


# In[12]:


y = lbl.fit_transform(y)
y


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[15]:


from sklearn.linear_model import LogisticRegression


# In[16]:


lgmodel = LogisticRegression()


# In[17]:


lgmodel.fit(x_train, y_train)


# In[18]:


y_pdt = lgmodel.predict(x_test)


# In[19]:


from sklearn.metrics import confusion_matrix


# In[20]:


confusion_matrix(y_test, y_pdt)


# In[21]:


28/30


# In[ ]:




