from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow.python.keras.engine import training
from tensorflow.python.ops.gradients_impl import gradients
from data_acquisition import get_data
from resnet import resnet_18
import hyper
import math

if __name__ == '__main__':
    model = resnet_18()
    model.build(input_shape=(None, hyper.IMAGE_SHAPE[0], hyper.IMAGE_SHAPE[1], hyper.IMAGE_SHAPE[2]))
    model.summary()

    train_dataset,train_count,test_dataset,test_count = get_data()

    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = hyper.scce_loss(y_true=labels,y_pred=predictions)
        gradients = tape.gradient(loss,model.trainable_variables)
        hyper.adadelta.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        hyper.mean_loss(loss)
        hyper.sccea_loss(labels,predictions)


    @tf.function
    def valid_step(images, labels):
        predictions = model(images, training=False)
        v_loss = hyper.scce_loss(labels, predictions)

        hyper.test_loss(v_loss)
        hyper.test_accuracy(labels, predictions)



    for epoch in range(hyper.EPOCHS):
        hyper.mean_loss.reset_states()
        hyper.sccea_loss.reset_states()
        step = 0
        for images, labels in train_dataset:
            step += 1
            train_step(images, labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                     hyper.EPOCHS,
                                                                                     step,
                                                                                     math.ceil(train_count / hyper.BATCH_SIZE),
                                                                                     hyper.mean_loss.result(),
                                                                                     hyper.sccea_loss.result()))
        for valid_images, valid_labels in test_dataset:
            valid_step(valid_images, valid_labels)

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                  hyper.EPOCHS,
                                                                  hyper.mean_loss.result(),
                                                                  hyper.sccea_loss.result(),
                                                                  hyper.test_loss.result(),
                                                                  hyper.test_accuracy.result()))
                                                                  