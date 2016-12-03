from __future__ import division

import os
import json
import argparse
import numpy as np
from os.path import join as pjoin

import nibabel as nib

from dipy.tracking.streamline import set_number_of_points
from dipy.segment.clustering import QuickBundles, QuickBundlesX, ClusterMapCentroid
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.distances import bundles_distances_mdf
from dipy.tracking.metrics import length as slength
from tractometer.io.streamlines import get_tracts_voxel_space_for_dipy

from scilpy.tractanalysis.robust_streamlines_metrics import compute_robust_tract_counts_map



def buildArgsParser():
    description = "Find optimal distance threshold for assignement."

    p = argparse.ArgumentParser(description=description)
    p.add_argument('full_tfile', help='tractogram file (.tck).')
    p.add_argument('model_tfile', help='tractogram file (.tck).')
    p.add_argument('model_mask', help='ground truth mask (.nii|.nii.gz).')
    p.add_argument('--qb-threshold', type=float, default=5,
                   help='QB threshold use to segment the model.')
    p.add_argument('--nb-points-resampling', type=int, default=12,
                   help='Number of points to use when resampling streamlines.')
    return p


def _compute_f1_score(overlap, overreach):
    # https://en.wikipedia.org/wiki/F1_score
    recall = overlap
    precision = 1 - overreach
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def _compute_overlap(basic_data, mask_data):
    basic_non_zero = np.count_nonzero(basic_data)
    overlap = np.logical_and(basic_data, mask_data)
    overlap_count = np.float32(np.count_nonzero(overlap))
    return overlap_count / basic_non_zero


def _compute_overreach(gt_data, candidate_data):
    diff = candidate_data - gt_data
    diff[diff < 0] = 0
    overreach_count = np.count_nonzero(diff)
    if np.count_nonzero(candidate_data) == 0:
        return 0

    return overreach_count / np.count_nonzero(candidate_data)


def _compute_overreach_normalize_gt(gt_data, candidate_data):
    diff = candidate_data - gt_data
    diff[diff < 0] = 0
    overreach_count = np.count_nonzero(diff)
    return overreach_count / np.count_nonzero(gt_data)


def create_binary_map(streamlines, ref_img):
    sl_map = compute_robust_tract_counts_map(streamlines, ref_img.shape)
    return (sl_map > 0).astype(np.int16)


def main():
    parser = buildArgsParser()
    args = parser.parse_args()

    full_tfile = nib.streamlines.load(args.full_tfile)
    model_tfile = nib.streamlines.load(args.model_tfile)
    model_mask = nib.load(args.model_mask)

    # Bring streamlines to voxel space and where coordinate (0,0,0) represents the corner of a voxel.
    model_tfile.tractogram.apply_affine(np.linalg.inv(model_mask.affine))
    model_tfile.streamlines._data += 0.5  # Shift of half a voxel
    full_tfile.tractogram.apply_affine(np.linalg.inv(model_mask.affine))
    full_tfile.streamlines._data += 0.5  # Shift of half a voxel

    assert(model_mask.get_data().sum() == create_binary_map(model_tfile.streamlines, model_mask).sum())

    # Resample streamlines
    full_streamlines = set_number_of_points(full_tfile.streamlines, args.nb_points_resampling)
    model_streamlines = set_number_of_points(model_tfile.streamlines, args.nb_points_resampling)

    # Segment model
    rng = np.random.RandomState(42)
    indices = np.arange(len(model_streamlines))
    rng.shuffle(indices)
    qb = QuickBundles(args.qb_threshold)
    clusters = qb.cluster(model_streamlines, ordering=indices)

    # Try to find optimal assignment threshold
    best_threshold = None
    best_f1_score = -np.inf
    thresholds = np.arange(-2, 10, 0.2) + args.qb_threshold
    for threshold in thresholds:
        indices = qb.find_closest(clusters, full_streamlines, threshold=threshold)
        nb_assignments = np.sum(indices != -1)

        mask = create_binary_map(full_tfile.streamlines[indices != -1], model_mask)

        overlap_per_bundle = _compute_overlap(model_mask.get_data(), mask)
        overreach_per_bundle = _compute_overreach(model_mask.get_data(), mask)
        # overreach_norm_gt_per_bundle = _compute_overreach_normalize_gt(model_mask.get_data(), mask)
        f1_score = _compute_f1_score(overlap_per_bundle, overreach_per_bundle)
        if best_f1_score < f1_score:
            best_threshold = threshold
            best_f1_score = f1_score

        print("{}:\t {}/{} ({:.1%}) {:.1%}/{:.1%} ({:.1%}) {}/{}".format(
            threshold,
            nb_assignments, len(model_streamlines), nb_assignments/len(model_streamlines),
            overlap_per_bundle, overreach_per_bundle, f1_score,
            mask.sum(), model_mask.get_data().sum()))

        if overlap_per_bundle >= 1:
            break


    print("Best threshold: {} with F1-Score of {}".format(best_threshold, best_f1_score))


if __name__ == "__main__":
    main()
