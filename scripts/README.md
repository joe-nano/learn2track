# Prepare HCP data

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



# Tractometer scoring (Python 2 only)

## Prerequisites
- TractConverter
`pip install https://github.com/MarcCote/tractconverter/archive/master.zip`

- Nibabel (Marc's bleeding_edge)
`pip install https://github.com/MarcCote/nibabel/archive/bleeding_edge.zip`

- texttable
`pip install texttable`

- Cython
`pip install cython`

- Dipy (Marc's bleeding_edge_for_learn2track)
`pip install https://github.com/MarcCote/dipy/archive/bleeding_edge_for_learn2track.zip`

- Scilpy found at https://bitbucket.org/MarcCote/scilpy (ask Marc)
`git clone https://MarcCote@bitbucket.org/MarcCote/scilpy.git`
`python setup.py build_no_gsl`
`pip install -e .`

- Tractometer found at https://bitbucket.org/MarcCote/tractometer/overview (ask Marc)
`git clone https://MarcCote@bitbucket.org/MarcCote/tractometer.git`
`cd tractometer`
`git checkout bleeding_edge`
`pip install -e .`


## Data

### ISMRM 2015 Challenge
Download the scoring data ``
More data available here `http://tractometer.org/ismrm_2015_challenge/data`

## Perform evaluation
`python ~/research/src/learn2track/scripts/score.py tractogram.tck scoring_data/ --out tractometer_folder --ismrm-tractometer`

## View score
`python ~/research/src/learn2track/scripts/tractometer/score_viewer.py --scores tractometer_folder/*/scores/*json`