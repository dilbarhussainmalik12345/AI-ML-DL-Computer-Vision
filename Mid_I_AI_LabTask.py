#!/usr/bin/env python
# coding: utf-8

# In[63]:


import numpy as np
import pandas as pd
import os
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[64]:


ds = pd.read_csv("house.csv")
ds


# In[92]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Area')
plt.xlabel('Bed')
plt.xlabel('Old')
plt.ylabel('Price')


# In[93]:


plt.scatter(ds.Area, ds.Price, color = 'red', marker = '*')


# In[94]:


new_ds = ds.drop('Price', axis = 'columns')
new_ds


# In[95]:


model = linear_model.LinearRegression()


# In[96]:


model.fit(new_ds, ds.Price)


# In[99]:


model.predict([[3200, 3,2]])


# In[ ]:




