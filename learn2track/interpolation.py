import numpy as np

import theano
import theano.tensor as T

# shape = (5, 5, 5)
# # I = np.ones(shape, dtype="float32")
# I = np.arange(int(np.prod(shape)), dtype="float32").reshape(shape)
# coords = np.array([[0, 0, 0], [1.5, .75, .5], [2, 0, .5], [3.25, 0, 1.25]]).astype("float32")
# values = map_coordinates(I, coords.T, mode="nearest", order=1)

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

# # P = I[np.floor(coords) + idx[:, None]]
# indices = (np.floor(coords)[:, None].astype(int) + idx.astype(int)).reshape((-1, 3))
# P = I[indices[:, 0], indices[:, 1], indices[:, 2]].reshape((len(coords), -1)).T
# # P = I[zip((np.floor(coords)[:, None] + idx).T)][0].T

# C = np.dot(B1, P)
# d = (coords - np.floor(coords)) / 1
# dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
# Q1 = np.c_[np.ones(len(d)), d, dx*dy, dy*dz, dx*dz, dx*dy*dz].T


# print(values)
# print(np.dot(C.T, Q1))
# print(np.dot(P.T, np.dot(B1.T, Q1)))
# print("")
# print(values-np.dot(C.T, Q1))
# print(values-np.sum(P * np.dot(B1.T, Q1), axis=0))


# coords2 = coords.copy()
# I2 = I

# coords = T.matrix("coords")
# coords.tag.test_value = coords2
# volume = T.tensor3("image")
# volume.tag.test_value = I2

# # indices = T.cast((T.floor(coords)[:, None, :] + idx).reshape((-1, 3)), dtype="int64")
# indices = T.cast((coords[:, None, :] + idx).reshape((-1, 3)), dtype="int32")
# P = volume[indices[:, 0], indices[:, 1], indices[:, 2]].reshape((coords.shape[0], -1)).T

# d = coords - T.floor(coords)
# dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
# Q1 = T.stack(T.ones_like(dx), d[:, 0], d[:, 1], d[:, 2], dx*dy, dy*dz, dx*dz, dx*dy*dz)
# V = T.sum(P * T.dot(B1.T, Q1), axis=0)
# f = theano.function([volume, coords], V)

# print(values-f(I2, coords2))

# B21 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
#                 [-1, 0, 0, 1, 0, 0, 0, 0],
#                 [0, 0, 0, 0, -1, 0, 1, 0],
#                 [-1, 1, 0, 0, 0, 0, 0, 0],
#                 [1, -1, 0, 0, -1, 1, 0, 0,],
#                 [0, 0, 0, 0, 1, -1, -1, 1]])

# Q2 = np.c_[np.ones(len(d)), dx*dy, dy*dz, dx*dz, dx*dy*dz].T

def advanced_indexing(volume, *indices_list, **kwargs):
    """
    Assuming `volume` is C contiguous.
    """
    strides = kwargs.get("strides")
    if strides is None:
        shapes = T.cast(volume.shape[:len(indices_list)], dtype=theano.config.floatX)
        strides = T.concatenate([T.ones((1,)), T.cumprod(shapes[::-1])[:-1]], axis=0)[::-1]

    indices = indices_list[-1]
    for i in range(len(indices_list)-1):
        indices += indices_list[i] * strides[i]

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
        indices = T.cast((coords[:, None, :] + idx).reshape((-1, 3)), dtype="int32")
        P = volume[indices[:, 0], indices[:, 1], indices[:, 2]].reshape((coords.shape[0], -1)).T

        d = coords - T.floor(coords)
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
        Q1 = T.stack([T.ones_like(dx), d[:, 0], d[:, 1], d[:, 2], dx*dy, dy*dz, dx*dz, dx*dy*dz], axis=0)
        values = T.sum(P * T.dot(B1.T, Q1), axis=0)
        return values

    elif volume.ndim == 4:
        # indices = T.cast((coords[:, None, :] + idx).reshape((-1, 3)), dtype="int32")
        # P = volume[indices[:, 0], indices[:, 1], indices[:, 2]].reshape((coords.shape[0], 8, -1)).T
        indices = T.floor((coords[:, None, :] + idx).reshape((-1, 3)))
        P = advanced_indexing(volume, indices[:, 0], indices[:, 1], indices[:, 2], strides=strides).reshape((coords.shape[0], 8, -1)).T

        d = coords - T.floor(coords)
        dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]

        # Unrolling
        Q1 = T.stack([T.ones_like(dx), d[:, 0], d[:, 1], d[:, 2], dx*dy, dy*dz, dx*dz, dx*dy*dz], axis=0)
        values = T.sum(P * T.dot(B1.T, Q1), axis=1).T
        return values
