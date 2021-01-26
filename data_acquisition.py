#loading dataset
import tensorflow as tf
from tensorflow import keras
import numpy as np
#import matplotlib.pyplot as plt
#from tensorflow._api.v2 import data


def cifar10():
    cifar = keras.datasets.cifar10 
    train, test = cifar.load_data()
    
    return train, test


@tf.autograph.experimental.do_not_convert
def data_preparation(train_tuple,test_tuple,train_batch=64,train_shuffle=10000,test_batch=5000,test_shuffle=10000):
    train_dataset = tf.data.Dataset.from_tensor_slices(train_tuple).batch(train_batch).shuffle(train_shuffle)
    train_dataset = train_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    #train_dataset = train_dataset.map(lambda x, y: (tf.image.central_crop(x, 0.75), y))
    #train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
    #train_dataset = train_dataset.repeat()
    train_dataset = tf.data.Dataset.zip(train_dataset)
    train_count = len(train_tuple[0])


    test_dataset = tf.data.Dataset.from_tensor_slices(test_tuple).batch(test_batch).shuffle(test_shuffle)
    test_dataset = test_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    #test_dataset = test_dataset.map(lambda x, y: (tf.image.central_crop(x, 0.75), y))
    #test_dataset = test_dataset.repeat()
    test_dataset = tf.data.Dataset.zip(test_dataset)
    test_count = len(test_tuple[0])

    return train_dataset, train_count, test_dataset, test_count


def build_fake_data(num_examples,IMAGE_SHAPE):
    x_train = np.random.rand(num_examples, *IMAGE_SHAPE).astype(np.float32)
    y_train = np.random.permutation(np.arange(num_examples)).astype(np.int32)
    x_test = np.random.rand(num_examples, *IMAGE_SHAPE).astype(np.float32)
    y_test = np.random.permutation(np.arange(num_examples)).astype(np.int32)
  
    return (x_train, y_train), (x_test, y_test)

def get_data():
    train, test = cifar10()
    train_dataset,train_count,test_dataset,test_count = data_preparation(train,test)
    
    return train_dataset,train_count,test_dataset,test_count