#!/usr/bin/env python
# coding: utf-8

# In[38]:


import tensorflow as tf


# In[39]:


from tensorflow import keras


# In[40]:


import matplotlib.pyplot as plt


# In[41]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[42]:


(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()


# In[43]:


len(X_train)


# In[44]:


len(X_test)


# In[45]:


X_train[0].shape


# In[46]:


X_train[0]


# In[47]:


plt.matshow(X_train[2])


# In[48]:


y_train[2]


# In[49]:


y_train[:5]


# In[50]:


X_train.shape


# In[51]:


X_train = X_train/255
X_test = X_test/255

X_train[0]
# In[54]:


X_train_flattened = X_train.reshape(len(X_train), 28*28)


# In[55]:


X_train_flattened


# In[56]:


X_train_flattened.shape


# In[57]:


X_test_flattened = X_test.reshape(len(X_test), 28*28)


# In[58]:


X_test_flattened.shape


# In[59]:


X_train_flattened[0]


# In[60]:


model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Corrected loss function name
    metrics=['accuracy']
)

model.fit(X_train_flattened, y_train, epochs=5)


# In[61]:


model.evaluate(X_test_flattened, y_test)


# In[69]:


plt.matshow(X_test[1])


# In[70]:


y_predicted = model.predict(X_test_flattened)
y_predicted[1]


# In[72]:


np.argmax(y_predicted[1])


# In[73]:


y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]


# In[74]:


y_test[:5]


# In[77]:


cm =tf.math.confusion_matrix(labels = y_test, predictions = y_predicted_labels)
cm


# In[78]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot = True, fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[80]:


model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Corrected loss function name
    metrics=['accuracy']
)

model.fit(X_train_flattened, y_train, epochs=5)


# In[81]:


model.evaluate(X_test_flattened, y_test)


# In[82]:


y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels = y_test, predictions = y_predicted_labels)


# In[83]:


plt.figure(figsize = (10,7))
sn.heatmap(cm, annot = True, fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[84]:


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Corrected loss function name
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=5)


# In[ ]:




