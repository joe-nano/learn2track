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
