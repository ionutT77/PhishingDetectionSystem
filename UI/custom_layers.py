"""Custom Keras layers used by the deployed model.

These exist to avoid using `keras.layers.Lambda` with Python bytecode serialization,
which is not portable across Python versions (and can crash the interpreter).
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="PhishingDetection")
class ReduceSum(keras.layers.Layer):
    def __init__(self, axis: int = 1, keepdims: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.keepdims = keepdims

    def call(self, inputs):  # type: ignore[override]
        return tf.reduce_sum(inputs, axis=self.axis, keepdims=self.keepdims)

    def compute_output_shape(self, input_shape):  # type: ignore[override]
        # input_shape: (..., steps, features)
        if input_shape is None:
            return None

        input_shape = tuple(input_shape)
        axis = self.axis
        if axis < 0:
            axis = len(input_shape) + axis

        if axis < 0 or axis >= len(input_shape):
            return input_shape

        if self.keepdims:
            output_shape = list(input_shape)
            output_shape[axis] = 1
            return tuple(output_shape)

        output_shape = list(input_shape)
        output_shape.pop(axis)
        return tuple(output_shape)

    def get_config(self):  # type: ignore[override]
        config = super().get_config()
        config.update({"axis": self.axis, "keepdims": self.keepdims})
        return config
