#%%
"""
Imports
"""
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import seaborn as sns
np.random.seed(0)

#%%
"""
Load Data
"""
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("Train Images: ", train_images.shape, "\nTrain Labels: ", train_labels.shape)
print("Test Images: ", test_images.shape, "\nTest Labels: ", test_labels.shape)

#%%
"""
Visualization of Data
"""
num_classes = 10
plot, subplots = plt.subplots(nrows=1, ncols=10, figsize=(20, 5))

for i in range(10):
    sample = train_images[train_labels == i][0]
    subplots[i].imshow(sample, cmap='gray')
    subplots[i].set_title("Label: {}".format(i), fontsize=16)

plt.show()

#%%
for i in range(num_classes):
    print(train_labels[i])

#%%
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

for i in range(num_classes):
    print(train_labels[i])

#%%
"""
Preparation of Data
"""
train_images = train_images / 255.0
test_images = test_images / 255.0


