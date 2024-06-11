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
Data Visualisation
"""
num_classes = 10
plot, subplots = plt.subplots(1, num_classes, figsize=(20, 5))

for i in range(num_classes):
    sample = train_images[train_labels == i][0]
    subplots[i].imshow(sample, cmap='gray')
    subplots[i].set_title("label: {}".format(i), fontsize=16)

plt.show()

#%%
for i in range(num_classes):
    print(train_labels[i])

#%%
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

for i in range(num_classes):
    print(test_labels[i])

#%%
"""
Data Preparation
"""
# Normalize data
train_images = train_images / 255.0
test_images = test_images / 255.0

#%%
# Reshape data
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)
print("Train Images: ", train_images.shape, "\nTrain Labels: ", train_labels.shape)
print("Test Images: ", test_images.shape, "\nTest Labels: ", test_labels.shape)


#%%
"""
Model creation
"""
model = Sequential()

#%%
model.add(Dense(units=128, input_shape=(784,), activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=10, activation='softmax'))

#%%
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#%%
# Train model
batch_size = 512
epochs = 10
model.fit(x=train_images, y=train_labels, batch_size=batch_size, epochs=epochs)

#%%
"""
Model Evaluation
"""
loss, accuracy = model.evaluate(test_images, test_labels)
print("Accuracy achieved: {}, Loss: {}".format(accuracy, loss))

#%%
model_output = model.predict(test_images)
print("Model Output: ", model_output)
#%%
predicted_classes = np.argmax(model_output, axis=1)
print("Predicted classes: ", predicted_classes)

#%%
# Single Example
random_index = np.random.choice(len(test_images))
sample_image = test_images[random_index]
true_test_labels = np.argmax(test_labels, axis=1)
sample_true_label = true_test_labels[random_index]
sample_predicted_label = predicted_classes[random_index]

#%%
plt.title("True label: {}, Predicted label: {}".format(sample_true_label, sample_predicted_label))
plt.imshow(sample_image.reshape(28, 28), cmap='gray')
plt.show()

#%%
"""
Confusion Matrix
"""
confusion_matrix = confusion_matrix(true_test_labels, predicted_classes)

plot, subplots = plt.subplots(figsize=(15, 10))
subplots = sns.heatmap(confusion_matrix, annot=True, fmt='d', ax=subplots, cmap='Blues')
subplots.set_xlabel('Predicted labels by model')
subplots.set_ylabel('True labels of images')
subplots.set_title('Confusion Matrix')
plt.show()

#%%
"""
Save Model
"""
model.save("mnist_model.h5")
