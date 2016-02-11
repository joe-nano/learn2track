#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import join as pjoin
from setuptools import setup, find_packages

setup(
    name='learn2track',',
    version='0.0.1',
    author='Marc-Alexandre Côté',
    author_email='marc-alexandre.cote@usherbrooke.ca',
    url='https://github.com/MarcCote/learn2track',
    packages=find_packages(),
    license='LICENSE',
    description='Learn how to do tractography directly from diffusion weighted images.',
    long_description=open('README.md').read(),
    install_requires=['smartlearner', 'dipy', 'nibabel'],
    dependency_links=['https://github.com/SMART-Lab/smartlearner/archive/master.zip#egg=smartlearner-0.0.1'],
    scripts=[pjoin('scripts', 'generate_training_data_ismrm2015.py'),
             pjoin('scripts', 'create_dataset_from_training_data.py'),
             pjoin('scripts', 'train_learn2track.py'),
             pjoin('scripts', 'use_learn2track.py')]
)
