# Prerequisite
Tractometer (ismrm_challenge_dev) found at https://bitbucket.org/MarcCote/tractometer/branch/ismrm_challenge_dev
Scilpy (challenge_scripts) found at https://bitbucket.org/jchoude/scilpy/branch/challenge_scripts

# Tracking
```
smart-dispatch -q qwork@mp2 -c 4 -t 24:00:00 launch source ~/env/p34/bin/activate \; python ~/research/src/learn2track/scripts/track.py [`find experiments/* -maxdepth 0 -type d -exec test -e "{}/GRU_Regression" -a ! -e "{}/tractometer/tracks/roi_2seeds_0.5mm.tck" ';' -print | xargs`] ismrm15_challenge/ground_truth/dwi.nii.gz --seeds ismrm15_challenge/scoring_data/masks/all_rois.nii.gz --mask ismrm15_challenge/ground_truth/wm.nii.gz --out roi_2seeds_0.5mm.tck --backward-tracking-algo 2 --nb-seeds-per-voxel 2 --step-size 0.5 --mask-threshold 0 --batch-size 10000
```

# Compute attributes for the Tractometer and move tractograms to ./{experiment_name}/tractometer/tracks
```
~/research/src/learn2track/scripts/tractometer/compute_attributes.sh experiments/*
```

# Generate command for Tractometer evaluation and execute them
```
python ~/research/src/learn2track/scripts/tractometer/gen_tractometer_commands.py experiments/*/tractometer/tracks/roi_2seeds_0.5mm.tck > scoring_roi_2seeds_0.5mm.cmd
```

smart-dispatch -q qwork@mp2 -c 4 -t 12:00:00 -f scoring_roi_2seeds_0.5mm.cmd launch

# Compute overreach and overlap
```
ipy-db ~/research/src/tractometer/scripts/scil_compute_bundle_overlap_overreach.py experiments/*/tractometer/scores/*.pkl ./ismrm15_challenge/scoring_data/masks/bundles/ -v
```

# Export results to a CSV file (Python 3)
```
python ~/research/src/learn2track/scripts/view_results.py experiments/* -v --tractography-names wm roi_2seeds_0.5mm
```
