#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np


# In[2]:


os.chdir("C:\\python")
os.getcwd()


# In[3]:


se_t1 = pd.read_csv("iris.csv")
se_t1


# In[4]:


x = se_t1.iloc[:, 0:4].values
x


# In[5]:


y = se_t1.iloc[:,4].values
y


# In[6]:


from sklearn.preprocessing import LabelEncoder


# In[7]:


l_b = LabelEncoder()


# In[8]:


y = l_b.fit_transform(y)
y


# In[9]:


from sklearn.model_selection import train_test_split


# In[21]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.5)


# In[22]:


from sklearn.ensemble import RandomForestClassifier


# In[23]:


classifier_rf = RandomForestClassifier(n_estimators = 3, criterion = 'entropy')


# In[24]:


classifier_rf.fit(x_train, y_train)


# In[25]:


y_Pred = classifier_rf.predict(x_test)


# In[26]:


from sklearn.metrics import confusion_matrix


# In[27]:


confusion_matrix(y_test, y_train)


# 26/83

# In[28]:


26/83


# In[ ]:




