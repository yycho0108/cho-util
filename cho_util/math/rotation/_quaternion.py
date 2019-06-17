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
    np.copysign(out[..., I_X], m21-m12, out=out[..., I_X])
    np.copysign(out[..., I_Y], m02-m20, out=out[..., I_Y])
    np.copysign(out[..., I_Z], m10-m01, out=out[..., I_Z])

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

    cc = cx*cz
    cs = cx*sz
    sc = sx*cz
    ss = sx*sz

    out[..., 0] = cy*sc - sy*cs
    out[..., 1] = cy*ss + sy*cc
    out[..., 2] = cy*cs - sy*sc
    out[..., 3] = cy*cc + sy*ss

    return out


def from_axis_angle(x, out=None):
    x = np.asarray(x)
    if out is None:
        out = np.empty(shape=np.shape(x)[:-1] + (4,))

    if x.shape[-1] == 3:
        # format : angle * axis
        angle = norm(x, keepdims=True)
        axis = x / angle
    elif x.shape[-1] == 4:
        # format : (axis, angle)
        axis = x[..., :3]
        angle = x[..., 3:]
    else:
        raise ValueError("Invalid Input Shape : {}".format(x.shape))
    half_angle = 0.5 * angle
    out[..., :3] = np.sin(half_angle) * axis
    out[..., 3:] = np.cos(half_angle)
    return out


def multiply(q1, q2, out=None):
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    if out is None:
        out = np.empty_like(q1)
    x1, y1, z1, w1 = [q1[..., i] for i in range(4)]
    x2, y2, z2, w2 = [q2[..., i] for i in range(4)]
    out[..., 0] = x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2
    out[..., 1] = -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2
    out[..., 2] = x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2
    out[..., 3] = -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2
    return out


def rotate(r, x, out=None):
    x = np.asarray(x)
    if out is None:
        out = np.empty_like(x)

    qx, qy, qz, qw = [r[..., i] for i in range(4)]
    x, y, z = [x[..., i] for i in range(3)]

    x0 = qw*x + qy*z - qz*y
    x1 = qx*x + qy*y + qz*z
    x2 = qw*z + qx*y - qy*x
    x3 = qw*y - qx*z + qz*x
    out[..., 0] = qw*x0 + qx*x1 + qy*x2 - qz*x3
    out[..., 1] = qw*x3 - qx*x2 + qy*x1 + qz*x0
    out[..., 2] = qw*x2 + qx*x3 - qy*x0 + qz*x1
    return out


def random(size, *args, **kwargs):
    size = tuple(np.reshape(size, [-1])) + (4,)
    out = np.random.normal(size=size, *args, **kwargs)
    out = uvec(out)
    return out