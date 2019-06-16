import numpy as np
from cho_util.math.common import *

from cho_util.math.rotation import _matrix
from cho_util.math.rotation import _quaternion
from cho_util.math.rotation import _euler
from cho_util.math.rotation import _axis_angle
from cho_util.math.rotation._euler import *


def to_matrix(x):
    return _matrix.from_euler(x)


def to_quaternion(x):
    return _quaternion.from_euler(x)


def to_euler(x):
    return _euler.from_euler(x)


def to_axis_angle(x):
    return _axis_angle.from_euler(x)
