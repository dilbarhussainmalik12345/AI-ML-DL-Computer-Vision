#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np


# In[3]:


os.chdir("C:\\python")
os.getcwd()


# In[4]:


set_d = pd.read_csv("iris.csv")
set_d


# In[6]:


x = set_d.iloc[:,:4].values
x


# In[7]:


y = set_d.iloc[:,4].values
y


# In[9]:


from sklearn.preprocessing import LabelEncoder


# In[10]:


lb_l = LabelEncoder()


# In[11]:


y = lb_l.fit_transform(y)
y


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


x_test, x_train, y_test, y_train = train_test_split(x,y,test_size = 0.2)


# In[14]:


from sklearn.svm import SVC


# In[15]:


classifier_svm_linear = SVC(kernel = 'linear')


# In[16]:


classifier_svm_linear.fit(x_train, y_train)


# In[17]:


y_Pred = classifier_svm_linear.predict(x_test)


# In[18]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_Pred)


# In[19]:


115/120


# In[ ]:




