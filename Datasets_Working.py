#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import pandas as pd
import numpy as np


# In[16]:


os.chdir("C:\\python\\")
os.getcwd()


# In[30]:


ds = pd.read_csv("abc.csv")
ds


# In[58]:


x=ds.iloc[:,0:4].values
x


# In[76]:


y=ds.iloc[:,4].values
y


# In[83]:


from sklearn.preprocessing import LabelEncoder


# In[84]:


le = LabelEncoder()


# In[85]:


y = le.fit_transform(y)


# In[86]:


y


# In[87]:


from sklearn.model_selection import train_test_split


# In[120]:


x_train, x_test, y_train,y_test = train_test_split(x,y,test_size = 0.2)


# In[121]:


from sklearn.linear_model import LogisticRegression


# In[122]:


lgm = LogisticRegression()


# In[123]:


lgm.fit(x_train, y_train)


# In[124]:


y_Pred = lgm.predict(x_test)


# In[125]:


from sklearn.metrics import confusion_matrix


# In[126]:


confusion_matrix(y_test, y_Pred)


# In[127]:


1/2


# In[ ]:




