import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = img_array.reshape(1, 784)
    img_array = img_array / 255.0
    return img_array

def predict_digit(image_path, model):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class

# Load trained model
model = load_model('mnist_model.h5')

# Provide path to the image
image_path = "mnist_image_2_2.png.jpg"

# Make prediction
predicted_digit = predict_digit(image_path, model)

# Display the image and the predicted digit
img = image.load_img(image_path, target_size=(28, 28), color_mode='grayscale')
plt.imshow(img, cmap='gray')
plt.title("Predicted Digit: {}".format(predicted_digit))
plt.axis('off')
plt.show()
