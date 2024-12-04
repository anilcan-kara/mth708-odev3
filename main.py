import tensorflow as tf
# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
# Print the shape of the training and test sets
print("Training set shape:", x_train.shape, y_train.shape)
print("Test set shape:", x_test.shape, y_test.shape)

import matplotlib.pyplot as plt
# Define class labels
class_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}
# Plot sample images from the training set
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_labels[y_train[i]])
plt.show()

# Data Exploration
import matplotlib.pyplot as plt
# Overview
print("Training set shape:", x_train.shape, y_train.shape)
print("Test set shape:", x_test.shape, y_test.shape)
# Example Images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_labels[y_train[i]])
plt.show()
# Data Preprocessing
# Normalization
x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0
# Reshaping for CNN
x_train_cnn = x_train_normalized.reshape((-1, 28, 28, 1))
x_test_cnn = x_test_normalized.reshape((-1, 28, 28, 1))
# Reshaping for RNN (assuming each row is a sequence)
x_train_rnn = x_train_normalized.reshape((-1, 28, 28))
x_test_rnn = x_test_normalized.reshape((-1, 28, 28))

from tensorflow.keras import layers, models
# Define CNN model
cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
# Compile CNN model
cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
# Train CNN model
cnn_history = cnn_model.fit(x_train_cnn, y_train, epochs=10, validation_data=(x_test_cnn, y_test))

# Define RNN model
rnn_model = models.Sequential([
    layers.SimpleRNN(128, input_shape=(28, 28)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
# Compile RNN model
rnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train RNN model
rnn_history = rnn_model.fit(x_train_rnn, y_train, epochs=10, validation_data=(x_test_rnn, y_test))

# Plot accuracy
plt.plot(cnn_history.history['accuracy'], label='CNN Train Accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='CNN Validation Accuracy')
plt.plot(rnn_history.history['accuracy'], label='RNN Train Accuracy')
plt.plot(rnn_history.history['val_accuracy'], label='RNN Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('CNN vs RNN Accuracy')
plt.show()
