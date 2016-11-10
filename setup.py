#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import join as pjoin
from setuptools import setup, find_packages

setup(
    name='learn2track',
    version='0.1.0',
    author='Marc-Alexandre Côté, Philippe Poulin',
    url='https://github.com/MarcCote/learn2track',
    packages=find_packages(),
    license='LICENSE',
    description='Learn to do tractography directly from diffusion weighted images.',
    long_description=open('README.md').read(),
    install_requires=['smartlearner', 'dipy', 'nibabel>=2.2.0.dev0'],
    dependency_links=['https://github.com/SMART-Lab/smartlearner/archive/master.zip#egg=smartlearner-0.0.1',
                      'https://github.com/MarcCote/nibabel/archive/bleeding_edge.zip#egg=nibabel-2.2.0.dev0'],
    scripts=[pjoin('scripts', 'process_streamlines.py'),
             pjoin('scripts', 'split_dataset.py'),
             pjoin('scripts', 'viz.py'),
             pjoin('scripts', 'learn.py'),
             pjoin('scripts', 'track.py')]
)
