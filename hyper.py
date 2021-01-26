from os import name
from tensorflow import keras

#define label
label = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
NUM_CLASSES = len(label)
IMAGE_SHAPE = [32, 32, 3]

# Experiment parameters
EPOCHS = 50
BATCH_SIZE = 128

# Loss function
mse_loss = keras.losses.MeanSquaredError()
scce_loss = keras.losses.SparseCategoricalCrossentropy()
mean_loss = keras.metrics.Mean(name='train_loss')
sccea_loss = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = keras.metrics.Mean(name='test_loss')
test_accuracy = keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# Optimizer
adadelta = keras.optimizers.Adadelta()
