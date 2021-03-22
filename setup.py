#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='distributedsearch',
      version='1.0',
      description='Distributed search using MPI',
      url='https://github.com/zengandrew/distributedsearch/',
      packages = find_packages(),
      install_requires = ['numpy', 'mpi4py']
)
