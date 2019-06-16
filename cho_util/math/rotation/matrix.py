import numpy as np
from cho_util.math.common import *

from cho_util.math.rotation import _matrix
from cho_util.math.rotation import _quaternion
from cho_util.math.rotation import _euler
from cho_util.math.rotation import _axis_angle
from cho_util.math.rotation._matrix import *


def to_matrix(x):
    return _matrix.from_matrix(x)


def to_quaternion(x):
    return _quaternion.from_matrix(x)


def to_euler(x):
    return _euler.from_matrix(x)


def to_axis_angle(x):
    return _axis_angle.from_matrix(x)


def main():
    q = np.random.normal(scale=10.0, size=(1000,4))
    q = uvec(q)
    m = _matrix.from_quaternion(q)
    q = _quaternion.from_matrix(m)
    print (q.shape)
    m_r = _matrix.from_quaternion(q)
    print (m.shape)
    print (m_r.shape)
    err = np.square(m - m_r).sum(axis=(-1,-2)) 
    argerr = np.argmax(err)
    print (argerr)
    print (err[argerr])
    print (m[argerr])
    print (m_r[argerr])


if __name__ == '__main__':
    main()
