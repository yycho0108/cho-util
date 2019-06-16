import numpy as np
from cho_util.math.common import *

def from_matrix(x, out=None):
    #TODO(yycho0108): investigate singularities
    #TODO(yycho0108): implement
    return out

def from_quaternion(x, out=None):
    #TODO(yycho0108): investigate singularities
    qw = x[..., 3]
    mag = np.sqrt(1.0 - qw*qw)
    out[..., :3] = x[..., :3] / mag
    out[..., 3] = 2 * np.arccos(qw)
    return out

def from_euler(x, out=None):
    #TODO(yycho0108): investigate singularities
    x = np.asarray(x)
    if out is None:
        out = np.empty(shape=np.shape(x)[:-1] + (4,))
    c, s = np.cos(x), np.sin(x)
    cx,cy,cz = [c[...,i] for i in range(3)]
    sx,sy,sz = [s[...,i] for i in range(3)]

    out[..., 0] = s1*s2*c3 + c1*c2*s3
    out[..., 1] = s1*c2*c3 + c1*s2*s3
    out[..., 2] = c1*s2*c3 - s1*c2*s3
    out[..., 3] = 2 * np.arccos(cx*cy*cz - sx*sy*sz)
    return out

def from_axis_angle(x, out=None):
    #TODO(yycho0108): investigate singularities
    x = np.asarray(x)
    if out is None:
        out = np.empty(shape=np.shape(x)[:-1] + (4,))
    np.copyto(out, x)
    return out
