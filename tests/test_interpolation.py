import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import numpy as np
import nibabel as nib
from os.path import join as pjoin
from numpy.testing import assert_array_almost_equal

import theano
import theano.tensor as T

from learn2track.interpolation import eval_volume_at_3d_coordinates_in_theano
from learn2track.neurotools import eval_volume_at_3d_coordinates


def test_trilinear_interpolation():
    trk = nib.streamlines.load(os.path.abspath(pjoin(__file__, '..', 'data', 'CA.trk')))
    trk.tractogram.apply_affine(np.linalg.inv(trk.affine))

    dwi = nib.load(os.path.abspath(pjoin(__file__, '..', 'data', 'dwi.nii.gz')))
    expected = eval_volume_at_3d_coordinates(dwi.get_data().astype('float32'), trk.streamlines._data)

    coords = T.matrix("coords")
    coords.tag.test_value = trk.streamlines._data
    volume = T.tensor3("image")
    volume.tag.test_value = dwi.get_data()[..., 0]
    fct = theano.function([volume, coords], eval_volume_at_3d_coordinates_in_theano(volume, coords))
    # theano.printing.pydotprint(fct, 'interpolation_vol3d', with_ids=True)

    # Process directly multiple 3D volumes then concatenate the results.
    values = []
    for i in range(dwi.shape[-1]):
        values_tmp = fct(dwi.get_data()[..., i], trk.streamlines._data)
        values.append(values_tmp)

    values = np.array(values).T
    assert_array_almost_equal(values, expected, decimal=4)

    # Process directly the 4D volume.
    volume = theano.shared(dwi.get_data())
    coords = theano.shared(trk.streamlines._data)

    # Precompute strides that will be used in the interpolation.
    shapes = T.cast(volume.shape[:-1], dtype=theano.config.floatX)
    strides = T.concatenate([T.ones((1,)), T.cumprod(shapes[::-1])[:-1]], axis=0)[::-1]
    volume_strides = strides.eval()

    values = eval_volume_at_3d_coordinates_in_theano(volume, coords, strides=volume_strides).eval()
    assert_array_almost_equal(values, expected, decimal=4)

    # fct = theano.function([], eval_volume_at_3d_coordinates_in_theano(volume, coords, strides=volume_strides))
    # theano.printing.pydotprint(fct, 'interpolation_vol4d', with_ids=True)

    # Test tahat coordinates outside the volume are clipped.
    coords = coords * np.max(dwi.shape).astype('float32')
    expected = eval_volume_at_3d_coordinates(dwi.get_data().astype('float32'), coords.eval())
    values = eval_volume_at_3d_coordinates_in_theano(volume, coords, strides=volume_strides).eval()
    assert_array_almost_equal(values, expected, decimal=4)

    coords = -coords
    expected = eval_volume_at_3d_coordinates(dwi.get_data().astype('float32'), coords.eval())
    values = eval_volume_at_3d_coordinates_in_theano(volume, coords, strides=volume_strides).eval()
    assert_array_almost_equal(values, expected, decimal=4)


test_trilinear_interpolation()
