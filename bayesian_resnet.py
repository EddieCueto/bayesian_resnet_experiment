# Copyright 2018 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Trains a Bayesian neural network to classify CIFAR-10 images.
The architecture can be either ResNet [1] or VGG [2].
To run with default arguments:
  ```
  bazel run tensorflow_probability/examples:cifar10_bnn
  ```
#### References
[1]: He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
     "Deep residual learning for image recognition."
     _Proceedings of the IEEE_, 2016.
     https://arxiv.org/abs/1512.03385
[2]: Simonyan, Karen, and Andrew Zisserman.
     "Very deep convolutional networks for large-scale image recognition."
     arXiv preprint arXiv:1409.1556 (2014).
     https://arxiv.org/pdf/1409.1556.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings

# Dependency imports
import matplotlib
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from probability.tensorflow_probability.examples.models.bayesian_resnet import bayesian_resnet
from probability.tensorflow_probability.examples.models.bayesian_vgg import bayesian_vgg

tf.compat.v1.disable_eager_execution()
matplotlib.use("Agg")
warnings.simplefilter(action="ignore")
tfd = tfp.distributions

IMAGE_SHAPE = [32, 32, 3]
learning_rate = 0.0001
epochs = 700
batch_size = 128
data_dir = os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "bayesian_neural_network/data")
model_dir = os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "bayesian_neural_network/")
eval_freq = 400
num_monte_carlo = 50
architecture = "resnet"
kernel_posterior_scale_mean = -9.0
kernel_posterior_scale_constraint = 0.2
kl_annealing = 50
subtract_pixel_mean = True
fake_data = False


def build_input_pipeline(x_train, x_test, y_train, y_test,
                         batch_size, valid_size):
  """Build an Iterator switching between train and heldout data."""

  x_train = x_train.astype("float32")
  x_test = x_test.astype("float32")

  x_train /= 255
  x_test /= 255

  y_train = y_train.flatten()
  y_test = y_test.flatten()

  if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

  print("x_train shape:" + str(x_train.shape))
  print(str(x_train.shape[0]) + " train samples")
  print(str(x_test.shape[0]) + " test samples")

  # Build an iterator over training batches.
  training_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, np.int32(y_train)))
  training_batches = training_dataset.shuffle(
      50000, reshuffle_each_iteration=True).repeat().batch(batch_size)
  training_iterator = tf.compat.v1.data.make_one_shot_iterator(training_batches)

  # Build a iterator over the heldout set with batch_size=heldout_size,
  # i.e., return the entire heldout set as a constant.
  heldout_dataset = tf.data.Dataset.from_tensor_slices(
      (x_test, np.int32(y_test)))
  heldout_batches = heldout_dataset.repeat().batch(valid_size)
  heldout_iterator = tf.compat.v1.data.make_one_shot_iterator(heldout_batches)

  # Combine these into a feedable iterator that can switch between training
  # and validation inputs.
  handle = tf.compat.v1.placeholder(tf.string, shape=[])
  feedable_iterator = tf.compat.v1.data.Iterator.from_string_handle(
      handle, training_batches.output_types, training_batches.output_shapes)
  images, labels = feedable_iterator.get_next()

  return images, labels, handle, training_iterator, heldout_iterator


def main():
  #del argv  # unused
  if tf.io.gfile.exists(model_dir):
    tf.compat.v1.logging.warning(
        "Warning: deleting old log directory at {}".format(model_dir))
    tf.io.gfile.rmtree(model_dir)
  tf.io.gfile.makedirs(model_dir)

  if fake_data:
    (x_train, y_train), (x_test, y_test) = build_fake_data()
  else:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

  (images, labels, handle, training_iterator, heldout_iterator) = build_input_pipeline(x_train, x_test, y_train, y_test, batch_size, 500)

  if architecture == "resnet":
    model_fn = bayesian_resnet
  else:
    model_fn = bayesian_vgg

  model = model_fn(
      IMAGE_SHAPE,
      num_classes=10,
      kernel_posterior_scale_mean=kernel_posterior_scale_mean,
      kernel_posterior_scale_constraint=kernel_posterior_scale_constraint)
  logits = model(images)
  labels_distribution = tfd.Categorical(logits=logits)

  # Perform KL annealing. The optimal number of annealing steps
  # depends on the dataset and architecture.
  t = tf.compat.v2.Variable(0.0)
  kl_regularizer = t / (kl_annealing * len(x_train) / batch_size)

  # Compute the -ELBO as the loss. The kl term is annealed from 0 to 1 over
  # the epochs specified by the kl_annealing flag.
  log_likelihood = labels_distribution.log_prob(labels)
  neg_log_likelihood = -tf.reduce_mean(input_tensor=log_likelihood)
  kl = sum(model.losses) / len(x_train) * tf.minimum(1.0, kl_regularizer)
  loss = neg_log_likelihood + kl

  # Build metrics for evaluation. Predictions are formed from a single forward
  # pass of the probabilistic layers. They are cheap but noisy
  # predictions.
  predictions = tf.argmax(input=logits, axis=1)
  with tf.compat.v1.name_scope("train"):
    train_accuracy, train_accuracy_update_op = tf.compat.v1.metrics.accuracy(
        labels=labels, predictions=predictions)
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate)
    train_op = opt.minimize(loss)
    update_step_op = tf.compat.v1.assign(t, t + 1)

  with tf.compat.v1.name_scope("valid"):
    valid_accuracy, valid_accuracy_update_op = tf.compat.v1.metrics.accuracy(
        labels=labels, predictions=predictions)

  init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                     tf.compat.v1.local_variables_initializer())

  stream_vars_valid = [
      v for v in tf.compat.v1.local_variables() if "valid/" in v.name
  ]
  reset_valid_op = tf.compat.v1.variables_initializer(stream_vars_valid)

  with tf.compat.v1.Session() as sess:
    sess.run(init_op)

    # Run the training loop
    train_handle = sess.run(training_iterator.string_handle())
    heldout_handle = sess.run(heldout_iterator.string_handle())
    training_steps = int(
        round(epochs * (len(x_train) / batch_size)))
    for step in range(training_steps):
      _ = sess.run([train_op,
                    train_accuracy_update_op,
                    update_step_op],
                   feed_dict={handle: train_handle})

      # Manually print the frequency
      if step % 100 == 0:
        loss_value, accuracy_value, kl_value = sess.run(
            [loss, train_accuracy, kl], feed_dict={handle: train_handle})
        print(
            "Step: {:>3d} Loss: {:.3f} Accuracy: {:.3f} KL: {:.3f}".format(
                step, loss_value, accuracy_value, kl_value))

      if (step + 1) % eval_freq == 0:
        # Compute log prob of heldout set by averaging draws from the model:
        # p(heldout | train) = int_model p(heldout|model) p(model|train)
        #                   ~= 1/n * sum_{i=1}^n p(heldout | model_i)
        # where model_i is a draw from the posterior
        # p(model|train).
        probs = np.asarray([sess.run((labels_distribution.probs),
                                     feed_dict={handle: heldout_handle})
                            for _ in range(num_monte_carlo)])
        mean_probs = np.mean(probs, axis=0)

        _, label_vals = sess.run(
            (images, labels), feed_dict={handle: heldout_handle})
        heldout_lp = np.mean(np.log(mean_probs[np.arange(mean_probs.shape[0]),
                                               label_vals.flatten()]))
        print(" ... Held-out nats: {:.3f}".format(heldout_lp))

        # Calculate validation accuracy
        for _ in range(20):
          sess.run(
              valid_accuracy_update_op, feed_dict={handle: heldout_handle})
        valid_value = sess.run(
            valid_accuracy, feed_dict={handle: heldout_handle})

        print(
            " ... Validation Accuracy: {:.3f}".format(valid_value))

        sess.run(reset_valid_op)


if __name__ == "__main__":
  #tf.compat.v1.app.run()
  main()