# Basic usage for training and tracking if you already have diffusion data and reference streamlines

## Training a model

The first step is to generate the necessary training files using process_streamlines.py:

`process_streamlines.py --out dataset.npz raw_signal diffusion.nii.gz --bvals diffusion.bvals --bvecs diffusion.bvecs bundle1.tck bundle2.tck`

This will give you the `dataset.npz` file that wraps all the necessary information for training, aka the diffusion, bvals/bvecs and the streamlines.

If you wish, you can split the dataset into training/validation/test sets using `split_dataset.py` :

`split_dataset.py dataset.npz --split 0.7 0.2 0.1`


This will output 3 files, namely `dataset_trainset.npz`, `dataset_validset.npz` and `dataset_testset.npz`.

You are then ready to train a model using a basic call to `learn.py`:


`learn.py --train-subjects dataset_trainset.npz --valid-subjects dataset_validset.npz --max-epoch 100 --lookahead 10  --batch-size 100 --Adam LR=0.002 --name experiment1 gru_regression --hidden-sizes 500 500`

Once that works, you can play around with the other options.

## Tracking with a trained model

Once you have a trained model, you will need the following before you are ready to track:

- The diffusion data and the bvals/bvecs with the same name as the diffusion (e.g. `diffusion.nii.gz`, `diffusion.bvals`, `diffusion.bvecs`)
- The binary tracking mask, you can keep it simple and use a WM mask (e.g. `wm.nii.gz`)
- The binary seeding mask, you need to choose whether you want to seed from all the WM (`wm.nii.gz`) or just the interface (`interface.nii.gz`)

`track.py --seeds wm.nii.gz --nb-seeds-per-voxel 1 --step-size 1 --mask wm.nii.gz --discard-stopped-by-curvature --theta 20 experiment1 diffusion.nii.gz`

Note that the step size is used to scale the length of the model's predicted direction; if not given, the model prediction will be used as is.



# Other instructions that might be helpful

## Prepare HCP data

HCP data are in LAS and currently our scripts only supports RAS. To convert them simply follow these steps.

## Prerequisites
Mrtrix 3.0 (https://github.com/MRtrix3/mrtrix3)
Scilpy found at https://bitbucket.org/sciludes/scilpy

Let's process HCP subject #100307
- `mkdir 100307`

Download its diffusion data (available on braindata for SCIL's members)
- `rsync $braindata/RawData/Human_Connectome_Project/minimal_process/100307_3T_Diffusion_preproc.zip 100307/`

Download the file `box.pkl` needed for cropping the volume and get only the brain.
- `rsync $braindata/ProcessedData/Human_Connectome_Project/Segmented_Bundles/CSD_prob_int_pft_fodf_tracts/HCP_Subjects/boxes/100307/box.new.pkl 100307/box.pkl`

Run the conversion script.
- `source las2ras_for_hcp.sh 100307`

If everything went right, clean temporary files.
- `source las2ras_for_hcp.sh 100307 clean`



## Tractometer scoring (Python 2 only)

### Prerequisites
- TractConverter
  - `pip install https://github.com/MarcCote/tractconverter/archive/master.zip`

- Nibabel (Marc's bleeding_edge)
  - `pip install https://github.com/MarcCote/nibabel/archive/bleeding_edge.zip`

- texttable
  - `pip install texttable`

- Cython
  - `pip install cython`

- Dipy (Marc's bleeding_edge_for_learn2track)
  - `pip install https://github.com/MarcCote/dipy/archive/bleeding_edge_for_learn2track.zip`

- Scilpy found at https://bitbucket.org/MarcCote/scilpy (ask Marc)
  - `git clone https://MarcCote@bitbucket.org/MarcCote/scilpy.git`
  - `python setup.py build_no_gsl`
  - `pip install -e .`

- Tractometer found at https://bitbucket.org/MarcCote/tractometer/overview (ask Marc)
  - `git clone https://MarcCote@bitbucket.org/MarcCote/tractometer.git`
  - `cd tractometer`
  - `git checkout bleeding_edge`
  - `pip install -e .`


### Data

#### ISMRM 2015 Challenge
Download the scoring data (More data available here `http://tractometer.org/ismrm_2015_challenge/data`)

### Perform evaluation
`python ~/research/src/learn2track/scripts/score.py tractogram.tck scoring_data/ --out tractometer_folder --ismrm-tractometer`

### View score
`python ~/research/src/learn2track/scripts/tractometer/score_viewer.py --scores tractometer_folder/*/scores/*json`
