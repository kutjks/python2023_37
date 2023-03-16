import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

mnist = keras.datasets.mnist
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()
plt.imshow(x_train_full[1])

x_train_norm = x_train_full/255.
x_test_norm = x_test/255.

x_valid, x_train = x_train_norm[:5000], x_train_norm[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

x_test = x_test_norm

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential()

model.add(keras.layers.Flatten(input_shape = [28,28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
model_histor = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid))

model.evaluate(x_test, y_test)

x_sample = x_test[:5]
y_probability = model.predict(x_sample)

y_probability.round()
#y_predict = model.predict_classes(x_sample)
#y_predict
