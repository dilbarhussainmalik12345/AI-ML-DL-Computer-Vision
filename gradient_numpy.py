# -*- coding: utf-8 -*-
"""Gradient numpy.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pdSBrqERDHF2p-ob1K2QyO-s28lgreMX
"""

import numpy as np

X = np.array([1,2,3,4], dtype = np.float32)
Y = np.array([2,4,6,8], dtype = np.float32)

w = 0.0

#Model prediction
def forward(x):
  return w * x

#Loss (Mean Square area)
def loss(y, ypred):
  return ((ypred - y)**2).mean()

#Gradient
def gradient(x,y,ypred):
  return np.dot(2*x, ypred-y).mean()

print(f'Prediction Before Training: f(5)= {forward(5):.3f}')

#Training
learning_rate = 0.01
n_iters = 1000
for epoch in range(n_iters):
  #prediction  = forward pass
  ypred = forward(X)

  #loss
  l = loss(Y, ypred)

  #gradients

  dw = gradient(X,Y,ypred)

  #Update the weights
  w -= learning_rate * dw
  if epoch % 10 == 0:
    print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction After Training: f(5)= {forward(5):.3f}')

