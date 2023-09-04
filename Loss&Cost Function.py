#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


y_predicted = np.array([1,1,0,0,1])
y_true = np.array([0.30, 0.7, 1,0,0.5])


# In[7]:


def mae(y_true, y_predicted):
    total_error = 0
    for yt, yp in zip(y_true, y_predicted):
        total_error += (yt - yp)
    print("Total Error", y_true)
    mae = total_error / len(y_true)
    print("MAE: ", mae)
    return mae


# In[8]:


mae(y_true, y_predicted)


# In[10]:


np.abs(y_predicted - y_true)


# In[12]:


np.mean(y_predicted - y_true)


# In[13]:


np.log([0.000])


# In[14]:


eplison = 1e-15


# In[18]:


y_predicted_new = [max(i, eplison) for i in y_predicted]


# In[19]:


y_predicted_new


# In[22]:


y_predicted_new = [min(i, 1-eplison) for i in y_predicted_new]


# In[23]:


y_predicted_new


# In[24]:


y_predicted_new = np.array(y_predicted_new)


# In[26]:


np.log(y_predicted_new)


# In[27]:


-np.mean(y_true*np.log(y_predicted_new) + (1-y_true)*(np.log(1-y_predicted_new)))


# In[ ]:




