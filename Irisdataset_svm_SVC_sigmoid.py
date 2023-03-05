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


dset = pd.read_csv("iris.csv")


# In[4]:


dset


# In[5]:


x = dset.iloc[:,:4].values
x


# In[6]:


y = dset.iloc[:,4].values
y


# In[7]:


from sklearn.preprocessing import LabelEncoder
lab_l = LabelEncoder()


# In[8]:


y = lab_l.fit_transform(y)
y


# In[9]:


from sklearn.model_selection import train_test_split


# In[11]:


x_test, x_train, y_test, y_train = train_test_split(x,y, test_size = 0.2)


# In[13]:


from sklearn.svm import SVC


# In[14]:


classifier_svm_sigmoid = SVC(kernel = 'sigmoid')
classifier_svm_sigmoid.fit(x_train, y_train)


# In[16]:


y_Pred = classifier_svm_sigmoid.predict(x_test)


# In[19]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_Pred)


# In[20]:


33/87


# In[ ]:




