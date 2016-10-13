#!/usr/bin/env python

import argparse

import nibabel as nb
import numpy as np
from numpy.lib.index_tricks import c_
import tractconverter as tc


def convert_HPC_to_t1(tracts_file_name, hcp_ref_img_fname, t1_ref_fname,
                      out_tracts_fname):

    hcp_img = nb.load(hcp_ref_img_fname)
    hcp_affine = hcp_img.get_affine()

    # Since the original tracts are in LPS space, flip the 2 first axes to have a LPS affine
    #hcp_affine[0,:] *= -1.0
    #hcp_affine[1,:] *= -1.0

    # Don't need to transpose.
    # hcp_affine = hcp_affine.T.astype('<f4')

    ref_img = nb.load(t1_ref_fname)
    lps_affine = ref_img.get_affine()

    # Same as before, bring to LPS space.
    #lps_affine[0,:] *= -1.0
    #lps_affine[1,:] *= -1.0

    # Transpose for speed
    lps_affine = lps_affine.T.astype('<f4')

    tract_format = tc.detect_format(tracts_file_name)
    tract_file = tract_format(tracts_file_name)

    out_f = tract_format.create(out_tracts_fname, tract_file.hdr)
    out_strl = []

    nb_strl = tract_file.hdr[0]
    nb_conv = 0

    # Equiv
    #orig_vol_corner = np.dot(hcp_affine, np.array([[0],[0],[0],[1]])) + np.array([[0.7/2.0],[0.7/2.0],[-0.7/2.0],[0]])
    orig_vol_corner = np.dot(hcp_affine, np.array([[-0.5],[-0.5],[-0.5],[1]])).T.flatten()

    v_shift_orig = (np.array([-0.7/2.0, -0.7/2.0, -0.7/2.0, 0.0]) - orig_vol_corner)[0:3]

    mins = []
    maxs = []

    for s in tract_file:
        # Bring to origin using only the translation, no rotation in matrix.
        grid_space = s + v_shift_orig

        # Flip because tracts seem to be flipped when inspected (even subject1_wmparc.nii.gz is.)
        flipped = (((grid_space + 0.7/2.0) - np.array([90, 108, 90])) * np.array([-1.0, -1.0, 1.0])) + np.array([90, 108, 90]) - 0.5
        out_strl.append(np.dot(c_[flipped, np.ones([flipped.shape[0], 1], dtype='<f4')], lps_affine)[:, :-1])

        mins.append(np.min(out_strl[-1], axis=0))
        maxs.append(np.max(out_strl[-1], axis=0))

        if len(out_strl) == 500000:
            nb_conv += 500000
            out_f += out_strl
            out_strl = []
            print('Converted {0} of {1}'.format(nb_conv, nb_strl))

    out_f += out_strl


def buildArgsParser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Convert GT streamlines from original HCP space to '
                    'generated T1 space.')
    p.add_argument('input', action='store',  metavar='INPUT', type=str,
                   help='Streamlines input file name in original HCP space.')
    p.add_argument('hcp_ref_img', action='store', metavar='HCP_REF', type=str,
                   help='Reference image in HCP space (classic: subject1_wmparc.nii.gz)')
    p.add_argument('t1_ref_img', action='store', metavar='T1_REF', type=str,
                   help='Reference image in T1 space (classic: T1.nii.gz)')
    p.add_argument('output', action='store',  metavar='OUTPUT', type=str,
                   help='Streamlines output file name, in generated T1 space')

    p.add_argument('-f', action='store_true', dest='isForce',
                   help='Force (overwrite output file). [%(default)s]')
    p.add_argument('-v', action='store_true', dest='isVerbose',
                   help='Produce verbose output. [%(default)s]')
    return p


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    convert_HPC_to_t1(args.input, args.hcp_ref_img, args.t1_ref_img, args.output)

if __name__ == "__main__":
    main()
