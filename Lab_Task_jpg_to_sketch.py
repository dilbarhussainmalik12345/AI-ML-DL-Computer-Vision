#!/usr/bin/env python
# coding: utf-8

# In[7]:


import cv2

# Load the image
img = cv2.imread('E:\\Memories\\HEC visit with Sir Muzamil\\131ND750\\sir.JPG')

# Convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Invert the grayscale image
inverted_image = 255 - gray_image

# Blur the inverted image using a Gaussian filter
blurred_image = cv2.GaussianBlur(inverted_image, (21, 21), 0)

# Blend the grayscale image with the blurred inverted image
sketch_image = cv2.divide(gray_image, 255 - blurred_image, scale=256)

# Save the sketch image
cv2.imwrite('E:\\Memories\\HEC visit with Sir Muzamil\\131ND750\\new1_sir.JPG', sketch_image)


# In[ ]:




