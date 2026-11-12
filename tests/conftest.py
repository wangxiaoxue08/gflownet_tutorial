"""Select CPU backends before tests import or initialize numerical libraries."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def pytest_configure() -> None:
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
