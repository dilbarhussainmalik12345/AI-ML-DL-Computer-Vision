#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math


# In[2]:


def sigmoid(x):
    return 1/(1 +math.exp(-x))


# In[3]:


sigmoid(100)


# In[4]:


sigmoid(1)


# In[5]:


sigmoid(-54)


# In[6]:


sigmoid(0.5)


# In[7]:


def tanh(x):
    return (math.exp(x) - math.exp(x) + math.exp(-x))


# In[9]:


tanh(56)


# In[10]:


tanh(50)


# In[11]:


def relu(x):
    return max(0,x)


# In[12]:


relu(-7)


# In[13]:


relu(1)


# In[14]:


relu(10)


# In[15]:


def leaky_relu(x):
    return max(0.1*x,x)


# In[16]:


leaky_relu(-100)


# In[17]:


leaky_relu(8)


# In[ ]:




