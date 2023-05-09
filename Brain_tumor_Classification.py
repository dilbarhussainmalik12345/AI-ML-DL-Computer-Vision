#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Load Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[23]:


#Prepare or Collect data
import os

path = os.listdir('C:/Users/Professor Jordan/Downloads/brain_tumor/Training')
classes = {'no_tumor': 0, 'pituitary_tumor': 1}


# In[27]:


import cv2
X = []
Y = []

for cls in classes:
    pth = 'C:/Users/Professor Jordan/Downloads/brain_tumor/Training/'+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+ '/' + j, 0)
        img = cv2.resize(img, (200, 200))
        X.append(img)
        Y.append(classes[cls])


# In[28]:


np.unique(Y)


# In[29]:


X = np.array(X)
Y = np.array(Y)


# In[32]:


pd.Series(Y).value_counts()


# In[33]:


X.shape


# In[34]:


#Visualize data
plt.imshow(X[0], cmap = 'gray')


# In[35]:


#Prepare data
X_updated = X.reshape(len(X), -1)
X_updated.shape


# In[36]:


#Split data
xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state = 10, test_size = .20)


# In[37]:


xtrain.shape, xtest.shape


# In[38]:


#Feature Scaling
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())


# In[39]:


xtrain = xtrain/255
xtest = xtest/255


# In[40]:


print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())


# In[41]:


#Feture Scaling
from sklearn.decomposition import PCA


# In[42]:


print(xtrain.shape, xtest.shape)


# In[47]:


pca = PCA(.98)

#pca_train = pca.fit_transform(xtrain)
#pca_test = pca.transform(xtest)

pca_train = xtrain
pca_test = xtest


# In[48]:


#print(pca_train.shape, pca_test.shape
#print(pca.n_components_)
#print(pca.n_features_)


# In[50]:


#Train Model
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[55]:


import warnings
warnings.filterwarnings('ignore')
lg = LogisticRegression(C = 0.1)
lg.fit(pca_train, ytrain)


# In[56]:


sv = SVC()
sv.fit(pca_train, ytrain)


# In[57]:


#Evaluation
print("Training Score: ", lg.score(pca_train, ytrain))
print("Testing Score: ", lg.score(pca_test, ytest))


# In[59]:


#Evaluation
print("Training Score: ", sv.score(pca_train, ytrain))
print("Testing Score: ", sv.score(pca_test, ytest))


# In[60]:


#Predictions
pred = sv.predict(pca_test)
np.where(ytest != pred)


# In[63]:


pred[36]


# In[64]:


ytest[36]


# In[65]:


#Test Model
dec = {0: 'No Tumor', 1: 'Positive'}


# In[74]:


plt.figure(figsize = (12, 8))
p = os.listdir('C:/Users/Professor Jordan/Downloads/brain_tumor/Testing/')
c = 1
for i in os.listdir('C:/Users/Professor Jordan/Downloads/brain_tumor/Testing/no_tumor/')[:9]:
    plt.subplot(3,3,c)
    
    img = cv2.imread('C:/Users/Professor Jordan/Downloads/brain_tumor/Testing/no_tumor/'+i, 0)
    img1 = cv2.resize(img, (200, 200))
    img1 = img1.reshape(1, -1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+= 1


# In[75]:


plt.figure(figsize = (12, 8))
p = os.listdir('C:/Users/Professor Jordan/Downloads/brain_tumor/Testing/')
c = 1
for i in os.listdir('C:/Users/Professor Jordan/Downloads/brain_tumor/Testing/pituitary_tumor/')[:16]:
    plt.subplot(4,4,c)
    
    img = cv2.imread('C:/Users/Professor Jordan/Downloads/brain_tumor/Testing/pituitary_tumor/'+i,0)
    img1 = cv2.resize(img, (200, 200))
    img1 = img1.reshape(1, -1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+= 1


# In[ ]:




