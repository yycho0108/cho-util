import numpy as np
from cho_util.math.common import *

from cho_util.math.rotation import _matrix
from cho_util.math.rotation import _quaternion
from cho_util.math.rotation import _euler
from cho_util.math.rotation import _axis_angle
from cho_util.math.rotation._axis_angle import *


def to_matrix(x):
    return _matrix.from_axis_angle(x)


def to_quaternion(x):
    return _quaternion.from_axis_angle(x)


def to_euler(x):
    return _euler.from_axis_angle(x)


def to_axis_angle(x):
    return _axis_angle.from_axis_angle(x)
