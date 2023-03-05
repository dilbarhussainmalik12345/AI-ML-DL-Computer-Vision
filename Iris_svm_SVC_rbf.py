#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np


# In[2]:


os.chdir("C:\\python")
os.getcwd()


# In[4]:


se_t = pd.read_csv("iris.csv")
se_t


# In[6]:


x = se_t.iloc[:,0:4].values
x


# In[7]:


y = se_t.iloc[:,4].values
y


# In[9]:


from sklearn.preprocessing import LabelEncoder
lbl_1 = LabelEncoder()


# In[10]:


y = lbl_1.fit_transform(y)
y


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


x_test, x_train, y_test, y_train = train_test_split(x,y, test_size = 0.2)


# In[13]:


from sklearn.svm import SVC
classifier_svm_rbf = SVC(kernel = 'rbf')
classifier_svm_rbf.fit(x_train, y_train)


# In[14]:


y_Pred = classifier_svm_rbf.predict(x_test)


# In[15]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_Pred)


# In[16]:


96/120


# In[ ]:




