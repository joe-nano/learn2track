#!/bin/bash

cd ./$1

if [ "$#" -eq 2 ]; then
    if [ "$2" = "clean" ]; then
        echo "Cleaning up"
        mv -f dwi_ras_brain_b1000.nii.gz dwi_ras_b1000.nii.gz
        mv -f dwi_ras_brain_b0.nii.gz dwi_ras_b0.nii.gz
        rm data.nii.gz bvals bvecs bvals_ras bvecs_ras data_ras.nii.gz
        rm -rI ./$1
    fi

else
    echo "Processing data"
    if [ ! -d ./$1 ]; then
        unzip $1_3T_Diffusion_preproc.zip
    fi

    mv $1/T1w/Diffusion/bvals .
    mv $1/T1w/Diffusion/bvecs .
    mv $1/T1w/Diffusion/data.nii.gz .
    mrconvert data.nii.gz data_ras.nii.gz -stride 1,2,3,4 -fslgrad bvecs bvals -export_grad_fsl bvecs_ras bvals_ras -force
    python ~/research/src/learn2track/scripts/extract_single_shell.py data_ras.nii.gz 1000 --bvals bvals_ras --bvecs bvecs_ras --basename dwi_ras
    # python ~/research/src/scilpy/scripts/apply_crop_bb.py dwi_ras_b0.nii.gz dwi_ras_brain_b0.nii.gz box.pkl -f
    # python ~/research/src/scilpy/scripts/apply_crop_bb.py dwi_ras_b1000.nii.gz dwi_ras_brain_b1000.nii.gz box.pkl -f
    python ~/research/src/scilpy/scripts/scil_crop_volume.py dwi_ras_b0.nii.gz dwi_ras_brain_b0.nii.gz --input_bbox box.pkl -f
    python ~/research/src/scilpy/scripts/scil_crop_volume.py dwi_ras_b1000.nii.gz dwi_ras_brain_b1000.nii.gz --input_bbox box.pkl -f

fi

cd -
