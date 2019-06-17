import numpy as np
from cho_util.math.common import *

from cho_util.math.rotation import _matrix
from cho_util.math.rotation import _quaternion
from cho_util.math.rotation import _euler
from cho_util.math.rotation import _axis_angle
from cho_util.math.rotation._matrix import *


def to_matrix(x, *args, **kwargs):
    return _matrix.from_matrix(x, *args, **kwargs)


def to_quaternion(x, *args, **kwargs):
    return _quaternion.from_matrix(x, *args, **kwargs)


def to_euler(x, *args, **kwargs):
    return _euler.from_matrix(x, *args, **kwargs)


def to_axis_angle(x, *args, **kwargs):
    return _axis_angle.from_matrix(x, *args, **kwargs)
