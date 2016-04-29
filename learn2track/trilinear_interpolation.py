import numpy as np

shape = (5, 5, 5)
# I = np.ones(shape, dtype="float32")
I = np.arange(int(np.prod(shape)), dtype="float32").reshape(shape)
coords = np.array([[0, 0, 0], [1.5, .75, .5], [2, 0, .5], [3.25, 0, 1.25]]).astype("float32")
values = map_coordinates(I, coords.T, mode="nearest", order=1)

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

# P = I[np.floor(coords) + idx[:, None]]
indices = (np.floor(coords)[:, None].astype(int) + idx.astype(int)).reshape((-1, 3))
P = I[indices[:, 0], indices[:, 1], indices[:, 2]].reshape((len(coords), -1)).T
# P = I[zip((np.floor(coords)[:, None] + idx).T)][0].T

C = np.dot(B1, P)
d = (coords - np.floor(coords)) / 1
dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
Q1 = np.c_[np.ones(len(d)), d, dx*dy, dy*dz, dx*dz, dx*dy*dz].T


print(values)
print(np.dot(C.T, Q1))
print(np.dot(P.T, np.dot(B1.T, Q1)))
print("")
print(values-np.dot(C.T, Q1))
print(values-np.sum(P * np.dot(B1.T, Q1), axis=0))


import theano
import theano.tensor as T

coords2 = coords.copy()
I2 = I

coords = T.matrix("coords")
coords.tag.test_value = coords2
I = T.tensor3("image")
I.tag.test_value = I2
# indices = T.cast((T.floor(coords)[:, None, :] + idx).reshape((-1, 3)), dtype="int64")
indices = T.cast((coords[:, None, :] + idx).reshape((-1, 3)), dtype="int32")
P = I[indices[:, 0], indices[:, 1], indices[:, 2]].reshape((coords.shape[0], -1)).T

d = coords - T.floor(coords)
dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
Q1 = T.stack(T.ones_like(dx), d[:, 0], d[:, 1], d[:, 2], dx*dy, dy*dz, dx*dz, dx*dy*dz)
V = T.sum(P * T.dot(B1.T, Q1), axis=0)
f = theano.function([I, coords], V)

print(values-f(I2, coords2))

# B21 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
#                 [-1, 0, 0, 1, 0, 0, 0, 0],
#                 [0, 0, 0, 0, -1, 0, 1, 0],
#                 [-1, 1, 0, 0, 0, 0, 0, 0],
#                 [1, -1, 0, 0, -1, 1, 0, 0,],
#                 [0, 0, 0, 0, 1, -1, -1, 1]])

# Q2 = np.c_[np.ones(len(d)), dx*dy, dy*dz, dx*dz, dx*dy*dz].T
