import numpy as np

import theano
import theano.tensor as T
from theano.ifelse import ifelse


B1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
               [-1, 0, 0, 0, 1, 0, 0, 0],
               [-1, 0, 1, 0, 0, 0, 0, 0],
               [-1, 1, 0, 0, 0, 0, 0, 0],
               [1, 0, -1, 0, -1, 0, 1, 0],
               [1, -1, -1, 1, 0, 0, 0, 0],
               [1, -1, 0, 0, -1, 1, 0, 0],
               [-1, 1, 1, -1, 1, -1, -1, 1]], dtype="float32")

idx = np.array([[0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1]], dtype="float32")


def advanced_indexing(volume, *indices_list, **kwargs):
    """ Performs advanced indexing on `volume`.

    This function exists because in Theano<=0.9 advanced indexing is
    only supported along the first dimension.

    Notes
    -----
    Assuming `volume` is C contiguous.
    """
    strides = kwargs.get("strides")
    if strides is None:
        shapes = T.cast(volume.shape[:len(indices_list)], dtype=theano.config.floatX)
        strides = T.concatenate([T.ones((1,)), T.cumprod(shapes[::-1])[:-1]], axis=0)[::-1]

    shapes = T.cast(volume.shape, dtype=theano.config.floatX)

    indices = T.maximum(0, T.minimum(indices_list[-1], shapes[len(indices_list)-1]-1))
    for i in range(len(indices_list)-1):
        clipped_idx = T.maximum(0, T.minimum(indices_list[i], shapes[i]-1))
        indices += clipped_idx * strides[i]

    # indices = T.sum(T.stack(indices_list, axis=1)*strides[:len(indices_list)], axis=1)
    indices = T.cast(indices, dtype="int32")
    return volume.reshape((-1, volume.shape[-1]))[indices]


def eval_volume_at_3d_coordinates_in_theano(volume, coords, strides=None):
    """ Evaluates the data volume at given coordinates using trilinear interpolation.

    This function is a Theano version of `learn2track.utils.eval_volume_at_3d_coordinates`.

    Parameters
    ----------
    volume : 3D array or 4D array
        Data volume.
    coords : ndarray of shape (N, 3)
        3D coordinates where to evaluate the volume data.
    strides : tuple
        Strides of the volume (for speedup). Default: detected automatically.

    References
    ----------
    [1] https://spie.org/samples/PM159.pdf
    """
    if volume.ndim == 3:
        print("eval_volume_at_3d_coordinates_in_theano with volume.ndim == 3 has not been tested.")
        indices = T.cast((coords[:, None, :] + idx).reshape((-1, 3)), dtype="int32")
        P = volume[indices[:, 0], indices[:, 1], indices[:, 2]].reshape((coords.shape[0], -1)).T
        # P = advanced_indexing(volume, indices[:, 0], indices[:, 1], indices[:, 2], strides=strides).reshape((coords.shape[0], -1)).T

        d = coords - T.floor(coords)
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        Q1 = T.stack([T.ones_like(dx), d[:, 0], d[:, 1], d[:, 2], dx*dy, dy*dz, dx*dz, dx*dy*dz], axis=0)
        values = T.sum(P * T.dot(B1.T, Q1), axis=0)
        return values

    elif volume.ndim == 4:
        indices = T.floor((coords[:, None, :] + idx).reshape((-1, 3)))
        P = advanced_indexing(volume, indices[:, 0], indices[:, 1], indices[:, 2], strides=strides).reshape((coords.shape[0], 8, -1)).T

        d = coords - T.floor(coords)
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        Q1 = T.stack([T.ones_like(dx), d[:, 0], d[:, 1], d[:, 2], dx*dy, dy*dz, dx*dz, dx*dy*dz], axis=0)
        values = T.sum(P * T.dot(B1.T, Q1), axis=1).T
        return ifelse(coords.shape[0] > 0, values, T.zeros((0, volume.shape[-1])))
