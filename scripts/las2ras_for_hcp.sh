#!/bin/bash
cd ./$1

if [ ! -d ./$1 ]; then
    unzip $1_3T_Diffusion_preproc.zip
fi

mv $1/T1w/Diffusion/bvals .
mv $1/T1w/Diffusion/bvecs .
mv $1/T1w/Diffusion/data.nii.gz .
~/research/src/mrtrix3/bin/mrconvert data.nii.gz data_ras.nii.gz -stride 1,2,3,4 -fslgrad bvecs bvals -export_grad_fsl bvecs_ras bvals_ras -force
python ~/research/src/learn2track/scripts/extract_single_shell.py data_ras.nii.gz 1000 --bvals bvals_ras --bvecs bvecs_ras --basename dwi_ras
python ~/research/src/scilpy/scripts/apply_crop_bb.py dwi_ras_b0.nii.gz dwi_ras_brain_b0.nii.gz box.pkl -f
python ~/research/src/scilpy/scripts/apply_crop_bb.py dwi_ras_b1000.nii.gz dwi_ras_brain_b1000.nii.gz box.pkl -f

# Cleaning up
mv -f dwi_ras_brain_b1000.nii.gz dwi_ras_b1000.nii.gz
mv -f dwi_ras_brain_b0.nii.gz dwi_ras_b0.nii.gz
rm -f data.nii.gz bvals bvecs
rm -f data_ras_b0.nii.gz bvals_ras bvecs_ras data_ras.nii.gz
rm -rf ./$1
