#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np
import cv2


# In[66]:


import PIL.Image as Image
import os


# In[67]:


import matplotlib.pylab as plt


# In[68]:


import tensorflow as tf


# In[69]:


import tensorflow_hub as hub


# In[70]:


from tensorflow import keras
from tensorflow.keras import layers


# In[71]:


from tensorflow.keras.models import Sequential


# In[72]:


IMAGE_SHAPE = (224,224)

classifier = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4", input_shape = IMAGE_SHAPE+(3,))
])


# In[73]:


gold_fish = Image.open('C://Users//Asad Computrs//Dilbar//goldfish.jpg').resize(IMAGE_SHAPE)
gold_fish


# In[74]:


gold_fish = np.array(gold_fish)/255.0
gold_fish.shape


# In[75]:


gold_fish[np.newaxis,...].shape


# In[76]:


result = classifier.predict(gold_fish[np.newaxis,...])
result.shape


# In[77]:


result


# In[78]:


predicted_label_index = np.argmax(result)
predicted_label_index


# In[79]:


image_labels = []
with open("C://Users//Asad Computrs//Dilbar//imagelebel.txt", "r") as f:
    image_labels = f.read().splitlines()
image_labels[:5]


# In[80]:


image_labels[predicted_label_index]


# In[81]:


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"


# In[82]:


data_dir = tf.keras.utils.get_file('flower_photos', origin = dataset_url, cache_dir = '.', untar = True)


# In[83]:


import pathlib 
data_dir = pathlib.Path(data_dir)
data_dir


# In[84]:


list(data_dir.glob('*/*.jpg'))[:5]


# In[85]:


image_count = len(list(data_dir.glob('*/*')))
print(image_count)


# In[86]:


roses = list(data_dir.glob('roses/*'))
roses[:5]


# In[87]:


Image.open(str(roses[3]))


# In[88]:


tulips = list(data_dir.glob('tulips/*'))
tulips


# In[89]:


Image.open(str(tulips[3]))


# In[90]:


flower_images_dict = {
    'roses': list(data_dir.glob('roses/*')),
    'daisy': list(data_dir.glob('daisy/*')),
    'dandelion': list(data_dir.glob('dandelion/*')),
    'sunflowers': list(data_dir.glob('sunflowers/*')),
    'tulips': list(data_dir.glob('tulips/*')),
}


# In[91]:


flower_labels_dict = {
    'roses': 0,
    'daisy': 1,
    'dandelion':2,
    'sunflowers': 3,
    'tulips': 4,
}


# In[92]:


str(flower_images_dict['roses'][0])


# In[93]:


img = cv2.imread(str(flower_images_dict['roses'][0]))


# In[94]:


img.shape


# In[95]:


cv2.resize(img, IMAGE_SHAPE).shape


# In[96]:


X,y = [], []
for flower_name, images in flower_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img, IMAGE_SHAPE)
        X.append(resized_img)
        y.append(flower_labels_dict[flower_name])


# In[97]:


X = np.array(X)
y = np.array(y)


# In[98]:


from sklearn.model_selection import train_test_split


# In[99]:


X_train, X_test, y_train,y_test = train_test_split(X,y, random_state = 0)


# In[100]:


X_train_scaled = X_train/255
X_test_scaled = X_test/255


# In[101]:


plt.axis('off')
plt.imshow(X[0])


# In[102]:


plt.axis('off')
plt.imshow(X[1])


# In[39]:


plt.axis('off')
plt.imshow(X[10])


# In[40]:


plt.axis('off')
plt.imshow(X[6])


# In[41]:


predicted = classifier.predict(np.array([X[0], X[1],X[2]]))
predicted = np.argmax(predicted, axis = 1)
predicted


# In[64]:


image_labels[795]


# In[43]:


image_labels[800]


# In[44]:


feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
pretrained_model_without_top_layer = hub.KerasLayer(
    feature_extractor_model, input_shape = (224,224, 3), trainable = False)


# In[45]:


num_of_flowers = 5
model = tf.keras.Sequential([
    pretrained_model_without_top_layer,
    tf.keras.layers.Dense(num_of_flowers)
])
model.summary()


# In[46]:


model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['acc'])

model.fit(X_train_scaled, y_train, epochs = 5)


# In[103]:


model.evaluate(X_test_scaled, y_test)


# In[ ]:




