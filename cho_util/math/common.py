import numpy as np


def rint(x):
    """ convert to rounded integer (int32) """
    return np.round(x).astype(np.int32)


def anorm(x):
    """ angular norm, convert to range [-pi,pi] """
    return (x + np.pi) % (2*np.pi) - np.pi


inv = np.linalg.inv


def norm(x, *args, **kwargs):
    if 'axis' not in kwargs:
        kwargs['axis'] = -1
    return np.linalg.norm(x, *args, **kwargs)


def uvec(x):
    """ convert to unit vector """
    n = norm(x, keepdims=True)
    return np.divide(x, n, out=np.zeros_like(x),
                     where=(n > np.finfo(np.float32).eps))
    #x / norm(x, keepdims=True)
    # if n < np.finfo(np.float32).eps:
    #    return x
    # else:
    #    return x / norm(x, keepdims=True)


def lerp(a, b, w):
    """ linear interpolation """
    return (a * (1.0-w)) + (b*w)
