#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

# Hack so you don't have to put the library containing this script in the PYTHONPATH.
sys.path = [os.path.abspath(os.path.join(__file__, '..', '..'))] + sys.path

import dipy
import numpy as np
import argparse

import nibabel as nib
from nibabel.streamlines import ArraySequence

from learn2track import utils
from learn2track.utils import save_bundle
from learn2track.utils import Timer, map_coordinates_3d_4d
from dipy.tracking.streamline import set_number_of_points

from dipy.data import get_sphere


def show_points_on_sphere(points, radius=1):
    import matplotlib.pyplot as plt
    from matplotlib import cm, colors
    from mpl_toolkits.mplot3d import Axes3D

    # Set colours and render
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-radius, radius])
    ax.set_ylim([-radius, radius])
    ax.set_zlim([-radius, radius])
    ax.set_aspect("equal")

    # Create a sphere
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    x = radius*np.sin(phi)*np.cos(theta)
    y = radius*np.sin(phi)*np.sin(theta)
    z = radius*np.cos(phi)
    ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)

    # Draw points
    xx, yy, zz = np.hsplit(np.asarray(points), 3)
    ax.scatter(xx, yy, zz, color="k", s=20)

    # Display
    plt.tight_layout()
    plt.show()


def show_dwi(weights):
    if weights.shape[-1] == 100:
        sphere = get_sphere('repulsion100')
    elif weights.shape[-1] == 724:
        sphere = get_sphere('repulsion724')

    from dipy.viz import fvtk
    r = fvtk.ren()
    sfu = fvtk.sphere_funcs(weights, sphere, scale=2.2, norm=True)
    sfu.RotateX(90)
    sfu.RotateY(180)
    fvtk.add(r, sfu)
    fvtk.show(r)


def build_classification_argparser(subparser):
    DESCRIPTION = "Generate training data for classfication tasks."
    p = subparser.add_parser("classification", description=DESCRIPTION, help=DESCRIPTION)
    p.add_argument('bundles', metavar='bundle', type=str, nargs="+", help='list of ground truth bundle files.')

    p.add_argument('--nb-directions', type=int, choices=[100, 784], default=100,
                   help="Number of directions (100 or 784) a streamline can take (i.e. nb. of classes). Default: 100")
    p.add_argument('--view-directions', action='store_true', help='Display the available directions.')
    p.add_argument('-v', '--verbose', action='store_true', help='enable verbose mode.')


def build_regression_argparser(subparser):
    DESCRIPTION = "Generate training data for regression tasks."
    p = subparser.add_parser("regression", description=DESCRIPTION, help=DESCRIPTION)
    p.add_argument('bundles', metavar='bundle', type=str, nargs="+", help='list of ground truth bundle files.')
    p.add_argument('-v', '--verbose', action='store_true', help='enable verbose mode.')


def buildArgsParser():
    DESCRIPTION = ("Script to generate training data from a list of streamlines bundle files."
                   " Each streamline is added twice: once flip (i.e. the points are reverse) and once as-is."
                   " Since we cannot predict the next direction for the last point we don't use it for training."
                   " The same thing goes for the last point in the reversed order.")
    p = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('dwi', type=str, help='Nifti file containing the original HCP dwi used in the ISMRM2015 challenge.')
    p.add_argument('--view-dwi', action='store_true', help='Display dwi in one voxel.')
    p.add_argument('--out', metavar='DIR', type=str, help='output folder where to put generated training data. Default: along the bundles')
    p.add_argument('--nb-points', type=int,
                   help="if specified, force all streamlines to have the same number of points. This is achieved using linear interpolation along the curve."
                        "Default: number of points per streamline is not modified.")

    p.add_argument('--sanity-check', action='store_true', help='perform sanity check on the data and quit.')
    p.add_argument('-v', '--verbose', action='store_true', help='enable verbose mode.')

    subparser = p.add_subparsers(title="Tasks", dest="tasks")
    subparser.required = True   # force 'required' testing
    build_regression_argparser(subparser)
    build_classification_argparser(subparser)

    return p


def process_data_for_regression(tractogram, weights, affine):
    """
    Generates the training data given a list of streamlines and some
    normalized diffusion weights.

    The training data consist of a list of $N$ sequences of $M_i$ inputs $x_ij$ and $M_i$
    targets $y_ij$, where $N$ is the number of streamlines and $M_i$ is the number of
    3D points of streamline $i$. The input $x_ij$ is a vector (32 dimensions) corresponding
    to the dwi data that have been trilinearly interpolated at the coordinate of the $j$-th
    point of the $i$-th streamline. The target $y_ij$ corresponds the (x, y, z) direction
    leading from the 3D point $j$ to the 3D point $j+1$ of streamline $i$.

    Parameters
    ----------
    tractogram : `nib.streamlines.Tractogram` object
        contains the points coordinates of N streamlines
    weights : 4D array
        normalized diffusion weights
    affine : ndarray shape (4, 4)
        affine vox->RAS+mm

    Returns
    -------
    inputs : `nib.streamlines.ArraySequence` object
        the interpolated dwi data for every 3D point of every streamline found in
        `tractogram.streamlines`.
    targets : `nib.streamlines.ArraySequence` object
        the direction leading from any 3D point to the next in every streamline.
    """

    inputs = []
    targets = []

    for i, streamline in enumerate(tractogram.streamlines):
        # Get diffusion weights for every points along the streamlines (the inputs).
        # The affine is provided in order to bring the streamlines back to voxel space.
        weights_interpolated = map_coordinates_3d_4d(weights, streamline, affine=affine)

        # We don't need the last point though (nothing to predict from it).
        inputs.append(weights_interpolated[:-1])
        # Also add the flip version of the streamline (except its last point).
        inputs.append(weights_interpolated[::-1][:-1])

        # Get streamlines directions (the targets)
        directions = streamline[1:, :] - streamline[:-1, :]
        targets.append(directions)
        targets.append(-directions)  # Flip directions

    return ArraySequence(inputs), ArraySequence(targets)


def process_data_for_classification(tractogram, weights, affine, sphere):
    """
    Generates the training data for classification task given a list of
    streamlines and some normalized diffusion weights.

    The training data consist of a list of $N$ sequences of $M_i$ inputs $x_ij$ and $M_i$
    targets $y_ij$, where $N$ is the number of streamlines and $M_i$ is the number of
    3D points of streamline $i$. The input $x_ij$ is a vector (32 dimensions) corresponding
    to the dwi data that have been trilinearly interpolated at the coordinate of the $j$-th
    point of the $i$-th streamline. The target $y_ij$ corresponds the direction ID
    leading from the 3D point $j$ to the 3D point $j+1$ of streamline $i$.

    Parameters
    ----------
    tractogram : `nib.streamlines.Tractogram` object
        contains the points coordinates of N streamlines
    weights : 4D array
        normalized diffusion weights
    affine : ndarray shape (4, 4)
        affine vox->RAS+mm
    sphere : `dipy.core.sphere.Sphere` object
        Allowed directions streamlines can move along (i.e. classes).

    Returns
    -------
    inputs : nib.streamlines.ArraySequence` object
        the interpolated dwi data for every 3D point of every streamline found in
        `tractogram.streamlines`.
    targets : `nib.streamlines.ArraySequence` object
        ID of the direction leading from any 3D point to the next in every streamline.
    """

    inputs = []
    targets = []

    for i, streamline in enumerate(tractogram.streamlines):
        # Get diffusion weights for every points along the streamlines (the inputs).
        # The affine is provided in order to bring the streamlines back to voxel space.
        weights_interpolated = map_coordinates_3d_4d(weights, streamline, affine=affine).astype(np.float32)
        inputs.append(weights_interpolated)
        directions = (streamline[1:, :] - streamline[:-1, :]).astype(np.float32)
        directions /= np.sqrt(np.sum(directions**2, axis=1, keepdims=True))
        targets.append(directions)
        #assert np.all(utils.find_closest(sphere, directions) == utils.find_closest(sphere, -directions))

        ## We don't need the last point though (nothing to predict from it).
        #inputs.append(weights_interpolated[:-1])
        ## Also add the flip version of the streamline (except its last point).
        #inputs.append(weights_interpolated[::-1][:-1])

        ## Get streamlines directions (the targets)
        #directions = streamline[1:, :] - streamline[:-1, :]
        #targets.append(directions)
        #targets.append(-directions)  # Flip directions
        ##targets.append(utils.find_closest(sphere, directions))
        ##targets.append(utils.find_closest(sphere, -directions))  # Flip directions

    inputs = ArraySequence(inputs)
    targets = ArraySequence(targets)
    #targets._data = utils.find_closest(sphere, targets._data).astype(np.int16)

    return inputs, targets


def do_sanity_check_on_data(streamlines, dwi):
    min_pts = np.inf*np.ones(3)
    max_pts = -np.inf*np.ones(3)
    for streamline_points in streamlines.points:
        min_pts = np.minimum(min_pts, streamline_points.min(axis=0))
        max_pts = np.maximum(max_pts, streamline_points.max(axis=0))

    if np.any(min_pts < np.zeros(3)-0.5) or np.any(max_pts >= np.array(dwi.shape[:3])+0.5):
        print ("Streamlines are not in voxel space!")
        print ("min_pts", min_pts)
        print ("max_pts", max_pts)
        return False

    return True


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    with Timer("Loading DWIs"):
        dwi = nib.load(args.dwi)

        # Load gradients table
        bvals_filename = args.dwi.split('.')[0] + ".bvals"
        bvecs_filename = args.dwi.split('.')[0] + ".bvecs"
        bvals, bvecs = dipy.io.gradients.read_bvals_bvecs(bvals_filename, bvecs_filename)

        dwi = nib.load(args.dwi)
        # weights = utils.normalize_dwi(dwi, bvals).astype(np.float32)
        weights = utils.resample_dwi(dwi, bvals, bvecs).astype(np.float32)  # Resample to 100 directions

        if args.view_dwi:
            # find an interesting voxel
            vox = np.unravel_index(np.argmax(weights.std(axis=3)), weights.shape[:3])
            show_dwi(weights[vox])

    if args.tasks == "classification":
        sphere = get_sphere("repulsion" + str(args.nb_directions))

        if args.view_directions:
            show_points_on_sphere(sphere.vertices)

    if args.sanity_check:
        for no, bundle in enumerate(args.bundles):
            streamlines = nib.streamlines.load(bundle, ref=dwi)
            print("Checking {0} ({1:,} streamlines)...".format(bundle, len(streamlines)), end=" ")
            is_ok = do_sanity_check_on_data(streamlines, dwi)
            print("OK" if is_ok else "FAIL")
    else:
        for bundle in args.bundles:
            if args.verbose:
                print("Processing {0}...".format(bundle))

            tractogram = nib.streamlines.load(bundle).tractogram  # Load streamlines in RASmm.

            if args.nb_points is not None:
                tractogram.streamlines = set_number_of_points(tractogram.streamlines, nb_points=args.nb_points)

            if args.tasks == "regression":
                inputs, targets = process_data_for_regression(tractogram, weights, dwi.affine)
            elif args.tasks == "classification":
                inputs, targets = process_data_for_classification(tractogram, weights, dwi.affine, sphere)

            # Dump data as numpy array
            if args.out is None:  # Put training data along the streamlines.
                path = os.path.splitext(bundle)[0] + ".npz"
            else:
                filename = os.path.splitext(os.path.basename(bundle))[0]
                path = os.path.join(args.out, filename + ".npz")

            save_bundle(path, inputs, targets)


if __name__ == '__main__':
    main()
