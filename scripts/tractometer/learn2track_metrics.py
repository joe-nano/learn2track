#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import logging
import os
import random

import nibabel as nib
import numpy as np
from dipy.segment.clustering import QuickBundles, QuickBundlesX
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.tracking.metrics import length as slength
from dipy.tracking.streamline import set_number_of_points
from tractometer.io.streamlines import get_tracts_voxel_space_for_dipy
from tractometer.metrics.invalid_connections import get_closest_roi_pairs_for_all_streamlines
from tractometer.utils.filenames import get_root_image_name

REF_BUNDLES_THRESHOLD = 5
NB_POINTS_RESAMPLE = 12


def _save_extracted_VBs(extracted_vb_info, streamlines,
                        segmented_out_dir, basename, ref_anat_fname):
    for bundle_name, bundle_info in extracted_vb_info.iteritems():
        if bundle_info['nb_streamlines'] > 0:
            out_fname = os.path.join(segmented_out_dir, basename +
                                     '_VB_{0}.tck'.format(bundle_name))

            # vb_f = TCK.create(out_fname)
            vc_strl = [streamlines[idx] for idx in bundle_info['streamlines_indices']]
            nib.streamlines.save(nib.streamlines.Tractogram(vc_strl, affine_to_rasmm=np.eye(4)), out_fname)
            # save_tracts_tck_from_dipy_voxel_space(vb_f, ref_anat_fname, vc_strl)


# From Max and Elef
# TODO after re-run: clean this
def auto_extract(model_cluster_map, rstreamlines,
                 number_pts_per_str=NB_POINTS_RESAMPLE,
                 close_centroids_thr=20,
                 clean_thr=7.,
                 disp=False, verbose=False,
                 ordering=None):
    if ordering is None:
        ordering = np.arange(len(rstreamlines))

    qb = QuickBundles(threshold=REF_BUNDLES_THRESHOLD, metric=AveragePointwiseEuclideanMetric())
    closest_bundles = qb.find_closest(model_cluster_map, rstreamlines, clean_thr, ordering=ordering)
    return ordering[np.where(closest_bundles >= 0)[0]]

    # model_centroids = model_cluster_map.centroids

    # centroid_matrix = bundles_distances_mdf(model_centroids,
    #                                         submission_cluster_map.centroids)

    # centroid_matrix[centroid_matrix > close_centroids_thr] = np.inf
    # mins = np.min(centroid_matrix, axis=0)
    # close_clusters = [submission_cluster_map[i] for i in np.where(mins != np.inf)[0]]
    # close_indices_inter = [submission_cluster_map[i].indices for i in np.where(mins != np.inf)[0]]
    # close_indices = list(chain.from_iterable(close_indices_inter))

    # close_streamlines = list(chain(*close_clusters))


    # closer_streamlines = close_streamlines
    # #matrix = np.eye(4)

    # rcloser_streamlines = set_number_of_points(closer_streamlines, number_pts_per_str)

    # clean_matrix = bundles_distances_mdf(model_cluster_map.refdata, rcloser_streamlines)

    # clean_matrix[clean_matrix > clean_thr] = np.inf

    # mins = np.min(clean_matrix, axis=0)
    # #close_clusters_clean = [closer_streamlines[i]
    # #                        for i in np.where(mins != np.inf)[0]]

    # clean_indices = [i for i in np.where(mins != np.inf)[0]]

    # # Clean indices refer to the streamlines in closer_streamlines,
    # # which are the same as the close_streamlines. Each close_streamline
    # # has a related element in close_indices, for which the value
    # # is the index of the original streamline in the moved_streamlines.
    # final_selected_indices = [close_indices[idx] for idx in clean_indices]

    # #return close_clusters_clean, final_selected_indices
    # return final_selected_indices


def _auto_extract_VCs(streamlines, ref_bundles):
    # Streamlines = list of all streamlines

    # TODO check what is neede
    # VC = 0
    VC_idx = set()

    found_vbs_info = {}
    for bundle in ref_bundles:
        found_vbs_info[bundle['name']] = {'nb_streamlines': 0,
                                          'streamlines_indices': set()}

    # TODO probably not needed
    # already_assigned_streamlines_idx = set()

    # Need to bookkeep because we chunk for big datasets
    processed_strl_count = 0
    chunk_size = len(streamlines)
    chunk_it = 0

    # nb_bundles = len(ref_bundles)
    # bundles_found = [False] * nb_bundles
    # bundles_potential_VCWP = [set()] * nb_bundles

    logging.debug("Starting scoring VCs")

    # Start loop here for big datasets
    while processed_strl_count < len(streamlines):
        if processed_strl_count > 0:
            raise NotImplementedError("Not supposed to have more than one chunk!")

        logging.debug("Starting chunk: {0}".format(chunk_it))

        strl_chunk = streamlines[chunk_it * chunk_size: (chunk_it + 1) * chunk_size]

        processed_strl_count += len(strl_chunk)

        # Already resample and run quickbundles on the submission chunk,
        # to avoid doing it at every call of auto_extract
        rstreamlines = set_number_of_points(nib.streamlines.ArraySequence(strl_chunk), NB_POINTS_RESAMPLE)

        # qb.cluster had problem with f8
        # rstreamlines = [s.astype('f4') for s in rstreamlines]

        # chunk_cluster_map = qb.cluster(rstreamlines)
        # chunk_cluster_map.refdata = strl_chunk

        # # Merge clusters
        # all_bundles = ClusterMapCentroid()
        # cluster_id_to_bundle_id = []
        # for bundle_idx, ref_bundle in enumerate(ref_bundles):
        #     clusters = ref_bundle["cluster_map"]
        #     cluster_id_to_bundle_id.extend([bundle_idx] * len(clusters))
        #     all_bundles.add_cluster(*clusters)

        # logging.debug("Starting VC identification through auto_extract")
        # qb = QuickBundles(threshold=10, metric=AveragePointwiseEuclideanMetric())
        # closest_bundles = qb.find_closest(all_bundles, rstreamlines, threshold=7)

        # print("Unassigned streamlines: {}".format(np.sum(closest_bundles == -1)))

        # for cluster_id, bundle_id in enumerate(cluster_id_to_bundle_id):
        #     indices = np.where(closest_bundles == cluster_id)[0]
        #     print("{}/{} ({}) Found {}".format(cluster_id, len(cluster_id_to_bundle_id), ref_bundles[bundle_id]['name'], len(indices)))
        #     if len(indices) == 0:
        #         continue

        #     vb_info = found_vbs_info.get(ref_bundles[bundle_id]['name'])
        #     indices = set(indices)
        #     vb_info['nb_streamlines'] += len(indices)
        #     vb_info['streamlines_indices'] |= indices
        #     VC_idx |= indices

        qb = QuickBundles(threshold=10, metric=AveragePointwiseEuclideanMetric())
        ordering = np.arange(len(rstreamlines))
        logging.debug("Starting VC identification through auto_extract")
        for bundle_idx, ref_bundle in enumerate(ref_bundles):
            print(ref_bundle['name'], ref_bundle['threshold'], len(ref_bundle['cluster_map']))
            # The selected indices are from [0, len(strl_chunk)]
            # selected_streamlines_indices = auto_extract(ref_bundle['cluster_map'],
            #                                             rstreamlines,
            #                                             clean_thr=ref_bundle['threshold'],
            #                                             ordering=ordering)

            closest_bundles = qb.find_closest(ref_bundle['cluster_map'], rstreamlines[ordering], ref_bundle['threshold'])
            selected_streamlines_indices = ordering[closest_bundles >= 0]
            ordering = ordering[closest_bundles == -1]

            # Remove duplicates, when streamlines are assigned to multiple VBs.
            # TODO better handling of this case
            # selected_streamlines_indices = set(selected_streamlines_indices) - cur_chunk_VC_idx
            # cur_chunk_VC_idx |= selected_streamlines_indices

            nb_selected_streamlines = len(selected_streamlines_indices)
            print("{} assigned".format(nb_selected_streamlines))

            if nb_selected_streamlines:
                # bundles_found[bundle_idx] = True
                # VC += nb_selected_streamlines

                # Shift indices to match the real number of streamlines
                global_select_strl_indices = set([v + chunk_it * chunk_size
                                                  for v in selected_streamlines_indices])
                vb_info = found_vbs_info.get(ref_bundle['name'])
                vb_info['nb_streamlines'] += nb_selected_streamlines
                vb_info['streamlines_indices'] |= global_select_strl_indices

                VC_idx |= global_select_strl_indices
                # already_assigned_streamlines_idx |= global_select_strl_indices

        chunk_it += 1

    return VC_idx, found_vbs_info


def score_auto_extract_auto_IBs(streamlines, bundles_masks, ref_bundles, ROIs, wm,
                                save_segmented=False, save_IBs=False,
                                save_VBs=False, save_VCWPs=False,
                                out_segmented_strl_dir='',
                                base_out_segmented_strl='',
                                ref_anat_fname=''):
    """
    TODO document


    Parameters
    ------------
    streamlines : sequence
        sequence of T streamlines. One streamline is an ndarray of shape (N, 3),
        where N is the number of points in that streamline, and
        ``streamlines[t][n]`` is the n-th point in the t-th streamline. Points
        are of form x, y, z in *voxel* coordinates.
    bundles_masks : sequence
        list of nibabel objects corresponding to mask of bundles
    ROIs : sequence
        list of nibabel objects corresponding to mask of ROIs
    wm : nibabel object
        mask of the white matter
    save_segmented : bool
        if true, returns indices of streamlines composing VC, IC, VCWP and NC

    Returns
    ---------
    scores : dict
        dictionnary containing a score for each metric
    indices : dict
        dictionnary containing the indices of streamlines composing VC, IC,
        VCWP and NC

    """

    # Load all streamlines, since streamlines is a generator.
    # full_strl = [s for s in streamlines]

    VC_indices, found_vbs_info = _auto_extract_VCs(streamlines, ref_bundles)
    VC = len(VC_indices)
    logging.debug('Found {} candidate VC'.format(VC))

    if save_VBs:
        _save_extracted_VBs(found_vbs_info, streamlines, out_segmented_strl_dir,
                            base_out_segmented_strl, ref_anat_fname)

    # TODO might be readded
    # To keep track of streamlines that have been classified
    # classified_streamlines_indices = VC_indices

    # New algorithm
    # Step 1: remove streamlines shorter than threshold (currently 35)
    # Step 2: apply Quickbundle with a hierarchical clustering based on the mean threshold + std of the reference bundles
    # Step 3: remove singletons
    # Step 4: assign to closest ROIs pair
    logging.debug("Starting IC, IB scoring")

    total_strl_count = len(streamlines)
    candidate_ic_strl_indices = sorted(set(range(total_strl_count)) - VC_indices)

    length_thres = 35.

    candidate_ic_streamlines = []
    rejected_streamlines = []

    for idx in candidate_ic_strl_indices:
        if slength(streamlines[idx]) >= length_thres:
            candidate_ic_streamlines.append(streamlines[idx].astype('f4'))
        else:
            rejected_streamlines.append(streamlines[idx].astype('f4'))

    logging.debug('Found {} candidate IC'.format(len(candidate_ic_streamlines)))
    logging.debug('Found {} streamlines that were too short'.format(len(rejected_streamlines)))

    ic_counts = 0
    ib_pairs = {}

    if len(candidate_ic_streamlines):

        # Fix seed to always generate the same output
        # Shuffle to try to reduce the ordering dependency for QB
        random.seed(0.2)
        random.shuffle(candidate_ic_streamlines)

        # Compute mean/std of bundles thresholds. Useful if using tractometer on different
        # environments like fibercup/ismrm/hcp
        thresholds_mean = np.mean([bundle['threshold'] for bundle in ref_bundles])
        thresholds_std = np.std([bundle['threshold'] for bundle in ref_bundles])
        thresholds = [thresholds_mean + 2 * thresholds_std, thresholds_mean, thresholds_mean - 2 * thresholds_std]

        logging.debug("Thresholds: {}".format(thresholds))

        qb = QuickBundlesX(thresholds)
        clusters_obj = qb.cluster(candidate_ic_streamlines)
        clusters = clusters_obj.get_clusters(-1)  # Retrieves clusters obtained with the smallest threshold.

        logging.debug("Found {} potential IB clusters".format(len(clusters)))

        # TODO this should be better handled
        rois_info = []
        for roi in ROIs:
            rois_info.append((get_root_image_name(os.path.basename(roi.get_filename())),
                              np.array(np.where(roi.get_data())).T))

        # TODO: Handle the case when streamlines are assigned to head and tail of the same bundle, i.e. a VC...

        centroids = nib.streamlines.Tractogram(clusters.centroids)
        centroids.apply_affine(np.linalg.inv(ROIs[0].affine))
        all_centroids_closest_pairs = get_closest_roi_pairs_for_all_streamlines(centroids.streamlines, rois_info)

        for c_idx, c in enumerate(clusters):
            closest_for_cluster = all_centroids_closest_pairs[c_idx]

            if closest_for_cluster not in ib_pairs:
                ib_pairs[closest_for_cluster] = []

            ic_counts += len(c)
            ib_pairs[closest_for_cluster].extend(c.indices)

        if save_segmented and save_IBs:
            for k, v in ib_pairs.iteritems():
                out_strl = np.array(candidate_ic_streamlines)[v]
                out_fname = os.path.join(out_segmented_strl_dir,
                                         base_out_segmented_strl + \
                                         '_IB_{0}_{1}.tck'.format(k[0], k[1]))

                nib.streamlines.save(nib.streamlines.Tractogram(out_strl, affine_to_rasmm=np.eye(4)), out_fname)

    if len(rejected_streamlines) > 0 and save_segmented:
        out_nc_fname = os.path.join(out_segmented_strl_dir,
                                    '{}_NC.tck'.format(base_out_segmented_strl))
        nib.streamlines.save(nib.streamlines.Tractogram(rejected_streamlines, affine_to_rasmm=np.eye(4)), out_nc_fname)

    # TODO readd classifed_steamlines_indices to validate
    if ic_counts != len(candidate_ic_strl_indices) - len(rejected_streamlines):
        raise ValueError("Some streamlines were not correctly assigned to NC")

    VC /= total_strl_count
    IC = (len(candidate_ic_strl_indices) - len(rejected_streamlines)) / total_strl_count
    NC = len(rejected_streamlines) / total_strl_count
    VCWP = 0

    # TODO could have sanity check on global extracted streamlines vs all
    # possible indices

    nb_VB_found = [v['nb_streamlines'] > 0 for k, v in found_vbs_info.iteritems()].count(True)
    streamlines_per_bundle = {k: v['nb_streamlines'] for k, v in found_vbs_info.iteritems() if v['nb_streamlines'] > 0}

    scores = {}
    scores['version'] = 2
    scores['algo_version'] = 5
    scores['VC'] = VC
    scores['IC'] = IC
    scores['VCWP'] = VCWP
    scores['NC'] = NC
    scores['VB'] = nb_VB_found
    scores['IB'] = len(ib_pairs.keys())
    scores['streamlines_per_bundle'] = streamlines_per_bundle
    scores['total_streamlines_count'] = total_strl_count

    return scores


def load_streamlines(filename, wm_file, tracts_attribs):
    try:
        tfile = nib.streamlines.load(filename)
        streamlines = tfile.tractogram.streamlines
    except:
        streamlines_gen = get_tracts_voxel_space_for_dipy(filename, wm_file, tracts_attribs)
        wm = nib.load(wm_file)
        tractogram = nib.streamlines.Tractogram(streamlines_gen, affine_to_rasmm=wm.affine)
        tractogram.to_world()
        streamlines = tractogram.streamlines

    return streamlines


def score_from_files(filename, masks_dir, bundles_dir,
                     tracts_attribs, basic_bundles_attribs,
                     save_segmented=False, save_IBs=False,
                     save_VBs=False, save_VCWPs=False,
                     segmented_out_dir='', segmented_base_name='',
                     verbose=False):
    """
    Computes all metrics in order to score a tractogram.

    Given a ``tck`` file of streamlines and a folder containing masks,
    compute the percent of: Valid Connections (VC), Invalid Connections (IC),
    Valid Connections but Wrong Path (VCWP), No Connections (NC),
    Average Bundle Coverage (ABC), Average ROIs Coverage (ARC),
    coverage per bundles and coverage per ROIs. It also provides the number of:
    Valid Bundles (VB), Invalid Bundles (IB) and streamlines per bundles.


    Parameters
    ------------
    filename : str
       name of a tracts file
    masks_dir : str
       name of the directory containing the masks
    save_segmented : bool
        if true, saves the segmented VC, IC, VCWP and NC

    Returns
    ---------
    scores : dict
        dictionnary containing a score for each metric
    indices : dict
        dictionnary containing the indices of streamlines composing VC, IC,
        VCWP and NC

    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    rois_dir = masks_dir + "rois/"
    bundles_masks_dir = masks_dir + "bundles/"
    wm_file = masks_dir + "wm.nii.gz"
    wm = nib.load(wm_file)

    streamlines = load_streamlines(filename, wm_file, tracts_attribs)

    ROIs = [nib.load(rois_dir + f) for f in sorted(os.listdir(rois_dir))]
    bundles_masks = [nib.load(bundles_masks_dir + f) for f in sorted(os.listdir(bundles_masks_dir))]
    ref_bundles = []

    # Ref bundles will contain {'name': 'name_of_the_bundle', 'threshold': thres_value,
    #                           'streamlines': list_of_streamlines}
    dummy_attribs = {'orientation': 'LPS'}

    out_centroids_dir = os.path.join(segmented_out_dir, os.path.pardir, "centroids")
    if not os.path.isdir(out_centroids_dir):
        os.mkdir(out_centroids_dir)

    rng = np.random.RandomState(42)

    for bundle_idx, bundle_f in enumerate(sorted(os.listdir(bundles_dir))):
        bundle_attribs = basic_bundles_attribs.get(os.path.basename(bundle_f))
        if bundle_attribs is None:
            raise ValueError("Missing basic bundle attribs for {0}".format(bundle_f))

        # Do not use a hardcoded threshold value for all bundles; instead, use a fraction of
        # the reference threshold for VC clustering
        # TODO: Implement a better way to do this; could be automatically computed based on number of streamlines + mean distance
        qb = QuickBundles(threshold=bundle_attribs['cluster_threshold'] / 2, metric=AveragePointwiseEuclideanMetric())

        orig_strl = load_streamlines(os.path.join(bundles_dir, bundle_f), wm_file, dummy_attribs)
        resamp_bundle = set_number_of_points(orig_strl, NB_POINTS_RESAMPLE)

        indices = np.arange(len(resamp_bundle))
        rng.shuffle(indices)
        bundle_cluster_map = qb.cluster(resamp_bundle, ordering=indices)

        # bundle_cluster_map.refdata = resamp_bundle

        bundle_mask_inv = nib.Nifti1Image((1 - bundles_masks[bundle_idx].get_data()) * wm.get_data(),
                                          bundles_masks[bundle_idx].get_affine())

        ref_bundles.append({'name': os.path.basename(bundle_f).replace('.fib', '').replace('.tck', ''),
                            'threshold': bundle_attribs['cluster_threshold'],
                            'cluster_map': bundle_cluster_map,
                            'mask': bundles_masks[bundle_idx],
                            'mask_inv': bundle_mask_inv})

        logging.debug("{}: {} centroids".format(ref_bundles[-1]['name'], len(bundle_cluster_map)))
        if verbose:
            print("Saving {} bundle centroids for {}".format(len(bundle_cluster_map.centroids), ref_bundles[-1]['name']))
            nib.streamlines.save(nib.streamlines.Tractogram(bundle_cluster_map.centroids, affine_to_rasmm=np.eye(4)),
                                 os.path.join(out_centroids_dir, ref_bundles[-1]['name'] + ".tck"))

    # Algorithm 5 is the only supported algorithm for now
    score_func = score_auto_extract_auto_IBs

    return score_func(streamlines, bundles_masks, ref_bundles, ROIs, wm,
                      save_segmented=save_segmented, save_IBs=save_IBs,
                      save_VBs=save_VBs, save_VCWPs=save_VCWPs,
                      out_segmented_strl_dir=segmented_out_dir,
                      base_out_segmented_strl=segmented_base_name,
                      ref_anat_fname=wm_file)
