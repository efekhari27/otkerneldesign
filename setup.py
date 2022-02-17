# coding: utf8
"""
Setup script for otkerneldesign
============================
This script allows to install otkerneldesign within the Python environment.
Usage
-----
::
    python setup.py install
"""
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='otkerneldesign',
    version='0.1',
    license='LGPL',
    author="Joesph Muré, Elias Fekhari",
    author_email='joesph.mure@edf.fr',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/efekhari27/otkerneldesign',
    keywords=['OpenTurns', 'KernelHerding'],
    description="Design of experiments based on kernel methods",
    install_requires=[
          "numpy",
          "scipy", 
          "openturns>=1.17"
      ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Natural Language :: French",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
    ],

)