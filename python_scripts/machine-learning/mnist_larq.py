#! usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import layers, models
import larq as lq
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Reshape to fit the model
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the binary neural network model
def build_binary_model():
    model = models.Sequential()

    breakpoint()
    # First binary convolutional layer
    model.add(
        lq.layers.QuantConv2D(
            32, kernel_size=3, strides=1, padding="same", input_shape=(28, 28, 1),
            kernel_quantizer="ste_sign", 
            input_quantizer="ste_sign", 
            use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Second binary convolutional layer
    model.add(
        lq.layers.QuantConv2D(
            64, kernel_size=3, strides=1, padding="same",
            kernel_quantizer="ste_sign", 
            input_quantizer="ste_sign", 
            use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Binary dense layer
    model.add(layers.Flatten())
    model.add(
        lq.layers.QuantDense(
            128,
            kernel_quantizer="ste_sign",
            input_quantizer="ste_sign",
            use_bias=False
        )
    )
    model.add(layers.BatchNormalization())

    # Output layer
    model.add(lq.layers.QuantDense(10, kernel_quantizer="ste_sign", input_quantizer="ste_sign", use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("softmax"))

    return model

model = build_binary_model()

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")

