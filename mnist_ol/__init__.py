import numpy as np
import tensorflow as tf

class MnistOL(object):
  def __init__(self, num_each_class=None):
    # train
    _train, _test = tf.keras.datasets.mnist.load_data()
    _mnist_train_x, _mnist_train_y = _train
    _mnist_train_y = _mnist_train_y.astype(np.int32)
    _mnist_train_x = (_mnist_train_x / 255).astype(np.float32)
    _mnist_train_i = np.arange(_mnist_train_x.shape[0]).astype(np.int32)
    _ordered = np.argsort(_mnist_train_y)
    _mnist_train_x = _mnist_train_x[_ordered]
    _mnist_train_y = _mnist_train_y[_ordered]
    if num_each_class is None:
      self._mnist_train_i = _mnist_train_i
      self._mnist_train_x = _mnist_train_x
      self._mnist_train_y = _mnist_train_y
    else:
      assert len(num_each_class) == 10
      _indices = [np.where(_mnist_train_y == i)[0][:num] for i, num in enumerate(num_each_class)]
      self._mnist_train_x = np.concatenate([_mnist_train_x[indice] for indice in _indices])
      self._mnist_train_y = np.concatenate([_mnist_train_y[indice] for indice in _indices])
      self._mnist_train_i = np.arange(self._mnist_train_x.shape[0]).astype(np.int32)

    _mnist_test_x, _mnist_test_y = _test
    self._mnist_test_y = _mnist_test_y.astype(np.int32)
    self._mnist_test_x = (_mnist_test_x / 255).astype(np.float32)
    self._mnist_test_i = np.arange(self._mnist_test_x.shape[0]).astype(np.int32)
    self._test_dataset = tf.data.Dataset.from_tensor_slices((
        self._mnist_test_i, self._mnist_test_x, self._mnist_test_y))

  @property
  def mnist_train_x(self):
    return self._mnist_train_x

  @property
  def mnist_train_y(self):
    return self._mnist_train_y

  def train_iterator(self, repeat=1):
    _train_dataset = tf.data.Dataset.from_tensor_slices((
        self._mnist_train_i, self._mnist_train_x, self._mnist_train_y))
    return _train_dataset.repeat(
        repeat).batch(1).prefetch(1000).make_one_shot_iterator()

  def test_iterator(self, batch):
    return self._test_dataset.repeat(1).batch(batch).prefetch(500).make_one_shot_iterator()
