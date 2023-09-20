#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 

np.set_printoptions(suppress=True, precision=6) 


# In[2]:


df = pd.read_csv("E:\\Semester-VII\\ML\\titanic.csv")
df.head()


# In[4]:


df.drop(["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin", "Embarked"], axis=1, inplace=True)
df.head()


# In[5]:


target = df["Survived"] 
inputs = df.drop("Survived", axis=1) 

display(target.head())
display(inputs.head())


# In[7]:


# converting the gender column into dummy variables
dum = pd.get_dummies(inputs["Sex"]) 

display(dum.head())
print(dum.dtypes)


# In[9]:


# concatenating the inputs dataframe with the dummies dataframe.
inputs = pd.concat([inputs, dum], axis=1)
inputs.head()


# In[10]:


# dropping the Sex column
#we add male and female coulmns
inputs.drop(["Sex"], axis=1, inplace=True)
inputs.head()


# In[12]:


inputs.Age[:10]


# In[14]:


# values to an integer type values.
inputs["Age"] = inputs["Age"].fillna(inputs["Age"].mean())

inputs.Age[:10]


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2)

print(len(X_train), len(X_test), len(inputs))


print(len(X_train) / len(inputs)) # training data %
print(len(X_test) / len(inputs)) # testing data %


# In[16]:


model = GaussianNB()
model.fit(X_train, y_train)


# In[17]:


model.score(X_test, y_test)


# In[ ]:





# In[ ]:




