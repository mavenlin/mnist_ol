import numpy as np
import tensorflow as tf

# train
_train, _test = tf.keras.datasets.mnist.load_data()
_mnist_train_x, _mnist_train_y = _train
_mnist_train_y = _mnist_train_y.astype(np.int32)
_mnist_train_x = (_mnist_train_x / 255).astype(np.float32)
_mnist_train_i = np.arange(_mnist_train_x.shape[0]).astype(np.int32)
_ordered = np.argsort(_mnist_train_y)
_mnist_train_x = _mnist_train_x[_ordered]
_mnist_train_y = _mnist_train_y[_ordered]
def train_iterator(repeat=1, num_each_class=None):
  if num_each_class is None:
    _mnist_train_i_subset = _mnist_train_i
    _mnist_train_x_subset = _mnist_train_x
    _mnist_train_y_subset = _mnist_train_y
  else:
    assert len(num_each_class) == 10
    _indices = [np.where(_mnist_train_y == i)[0][:num] for i, num in enumerate(num_each_class)]
    _mnist_train_x_subset = np.concatenate([_mnist_train_x[indice] for indice in _indices])
    _mnist_train_y_subset = np.concatenate([_mnist_train_y[indice] for indice in _indices])
    _mnist_train_i_subset = np.arange(_mnist_train_x_subset.shape[0]).astype(np.int32)
  _train_dataset = tf.data.Dataset.from_tensor_slices((
      _mnist_train_i_subset, _mnist_train_x_subset, _mnist_train_y_subset))
  return _train_dataset.repeat(
      repeat).batch(1).prefetch(1000).make_one_shot_iterator()


# test
_mnist_test_x, _mnist_test_y = _test
_mnist_test_y = _mnist_test_y.astype(np.int32)
_mnist_test_x = (_mnist_test_x / 255).astype(np.float32)
_mnist_test_i = np.arange(_mnist_test_x.shape[0]).astype(np.int32)
_test_dataset = tf.data.Dataset.from_tensor_slices((
    _mnist_test_i, _mnist_test_x, _mnist_test_y))
def test_iterator(batch):
  return _test_dataset.repeat(1).batch(batch).prefetch(500).make_one_shot_iterator()
