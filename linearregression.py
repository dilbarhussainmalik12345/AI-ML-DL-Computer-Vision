# -*- coding: utf-8 -*-
"""LinearRegression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yx9H4gL864iVAiLL36gIzpNgXccuS7pj
"""

# 1.Design Model(input, outputsize, forward pass)
# 2. Construct Loss and Optimizer
# 3. Training Loop
#   - Forwards pass: ComputePrediction
#   - Backwards Pass: Gradient
#  - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 0) Prepare data
X_numpy, y_numpy = datasets.make_regression(n_samples = 100, n_features=1, noise = 20, random_state=1)
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1)
n_samples, n_features = X.shape

# 1)Model
input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)

# 2)Loss and Optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 3)Training Loop
num_epochs = 100
for epoch in range(num_epochs):
  #Forwards pass and loss
   y_predicted = model(X)
   loss = criterion(y_predicted, y)

  #backwards pass
   loss.backward()

  #Update
   optimizer.step()
   optimizer.zero_grad()

   if (epoch+1) % 10 == 0:
      print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')


#Plot

predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show()

