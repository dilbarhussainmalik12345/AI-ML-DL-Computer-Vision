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


set1 = pd.read_csv("iris.csv")
set1


# In[4]:


x = set1.iloc[:, 0:4].values
x


# In[6]:


y = set1.iloc[:,4].values
y


# In[9]:


from sklearn.preprocessing import LabelEncoder


# In[10]:


lb_E = LabelEncoder()


# In[11]:


y = lb_E.fit_transform(y)
y


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


x_test, x_train, y_test, y_train = train_test_split(x,y,test_size = 0.2)


# In[14]:


from sklearn.svm import SVC
classifier_svm_poly = SVC(kernel = 'poly')


# In[15]:


classifier_svm_poly.fit(x_train, y_train)


# In[16]:


y_Pred = classifier_svm_poly.predict(x_test)


# In[17]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_Pred)


# In[18]:


115/120


# In[ ]:




