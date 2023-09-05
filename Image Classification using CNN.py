#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


import numpy as np


# In[13]:


(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()


# In[14]:


X_train.shape


# In[15]:


X_test.shape


# In[19]:


y_train[:5]


# In[21]:


y_train = y_train.reshape(-1,)
y_train[:5]


# In[28]:


classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship","truck"]


# In[29]:


classes[9]


# In[30]:


def plot_sample(X,y,index):
    
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


# In[31]:


plot_sample(X_train, y_train, 0)


# In[33]:


plot_sample(X_train, y_train, 1)


# In[34]:


plot_sample(X_train, y_train, 2)


# In[35]:


plot_sample(X_train, y_train, 3)


# In[38]:


X_train = X_train/255
X_test = X_test/255


# In[39]:


ann = models.Sequential([
    layers.Flatten(input_shape = (32,32,3)),
    layers.Dense(3000, activation = 'relu'),
    layers.Dense(1000, activation = 'relu'),
    layers.Dense(10, activation = 'sigmoid'),
])

ann.compile(optimizer = 'SGD',
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])
ann.fit(X_train, y_train, epochs = 5)


# In[40]:


ann.evaluate(X_test, y_test)


# In[42]:


from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))


# In[50]:


cnn = models.Sequential([
    
    # CNN layers
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[51]:


cnn.compile(optimizer = 'adam',
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])


# In[52]:


cnn.fit(X_train, y_train, epochs = 10)


# In[53]:


cnn.evaluate(X_test, y_test)


# In[63]:


y_test = y_test.reshape(-1,)
y_test[:5]


# In[64]:


plot_sample(X_test, y_test,1)


# In[66]:


y_pred = cnn.predict(X_test)
y_pred[:5]


# In[69]:


y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


# In[70]:


y_test[:5]


# In[81]:


plot_sample(X_test, y_test,3)


# In[82]:


classes


# In[84]:


classes[y_classes[3]]


# In[ ]:




