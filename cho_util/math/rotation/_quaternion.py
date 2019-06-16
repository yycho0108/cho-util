import numpy as np
from cho_util.math.common import *

I_X = 0
I_Y = 1
I_Z = 2
I_W = 3

def from_matrix(x, out=None):
    x = np.asarray(x)
    if out is None:
        out = np.empty(shape=np.shape(x)[:-2] + (4,))

    # parse input
    m00, m01, m02 = [x[..., 0, i] for i in range(3)]
    m10, m11, m12 = [x[..., 1, i] for i in range(3)]
    m20, m21, m22 = [x[..., 2, i] for i in range(3)]

    out[..., I_X] = 1 + m00 - m11 - m22
    out[..., I_Y] = 1 - m00 + m11 - m22
    out[..., I_Z] = 1 - m00 - m11 + m22
    out[..., I_W] = 1 + m00 + m11 + m22

    # sqrt(max(0,x))/2
    for i in range(4):
        np.maximum(0.0, out[..., i], out=out[..., i])
    np.sqrt(out, out=out)
    out *= 0.5

    # correct for signs
    np.copysign(out[..., I_X], m21-m12, out=out[...,I_X])
    np.copysign(out[..., I_Y], m02-m20, out=out[...,I_Y])
    np.copysign(out[..., I_Z], m10-m01, out=out[...,I_Z])

    return out

def from_quaternion(x, out=None):
    x = np.asarray(x)
    if out is None:
        out = np.empty(shape=np.shape(x)[:-1] + (4,))
    np.copyto(out, x)
    return out


def from_euler(x, out=None):
    x = np.asarray(x)
    if out is None:
        out = np.empty(shape=np.shape(x)[:-1] + (4,))
    half_x = (0.5 * x)
    c, s = np.cos(half_x), np.sin(half_x)

    cx, cy, cz = [c[..., i] for i in range(3)]
    sx, sy, sz = [s[..., i] for i in range(3)]
    out[..., 0] = cx * cy * cz + sx * sy * sz
    out[..., 1] = -cx * sy * sz + sx * cy * cz
    out[..., 2] = cx * sy * cz + sx * cy * sz
    out[..., 3] = -sx * sy * cz + cx * cy * sz
    return out


def from_axis_angle(x, out=None):
    x = np.asarray(x)
    if out is None:
        out = np.empty(shape=np.shape(x)[:-1] + (4,))

    if x.shape[-1] == 3:
        # format : angle * axis
        angle = norm(x)
        axis = x / angle[..., None]
    elif x.shape[-1] == 4:
        # format : (axis, angle)
        axis = x[..., :3]
        angle = x[..., 3]
    else:
        raise ValueError("Invalid Input Shape : {}".format(x.shape))
    half_angle = 0.5 * angle
    out[..., 0] = np.cos(half_angle)
    out[..., 1:] = np.sin(half_angle) * axis
    return out
