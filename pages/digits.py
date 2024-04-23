import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

# Function to create the model architecture
def create_model():
    model = Sequential([
        Conv2D(28, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model

# Load or train the MNIST model
model_file = 'mnist_model.h5'
mnist_model = create_model()
try:
    mnist_model.load_weights(model_file)
except:
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, -1) / 255.0
    x_test = np.expand_dims(x_test, -1) / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Compile and train the model
    mnist_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    mnist_model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    # Save the model weights for future use
    mnist_model.save_weights(model_file)

def predict_digit(img):
    img = ImageOps.grayscale(img.resize((28, 28)))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)  # Reshape for the model
    prediction = mnist_model.predict(img)
    return np.argmax(prediction)

# Streamlit UI
st.title("Handwritten Digit Recognition")
st.write("This app predicts the digit you've written.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Predicting...")
    label = predict_digit(image)
    st.write(f"The handwritten digit is: {label}")
