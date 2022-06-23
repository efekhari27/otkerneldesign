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

import re
import os
from setuptools import setup, find_packages

# Get the version from __init__.py
path = os.path.join(os.path.dirname(__file__), 'otkerneldesign', '__init__.py')
with open(path) as f:
    version_file = f.read()

version = re.search(r"^\s*__version__\s*=\s*['\"]([^'\"]+)['\"]",
                    version_file, re.M)
if version:
    version = version.group(1)
else:
    raise RuntimeError("Unable to find version string.")

# Long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='otkerneldesign',
    version=version,
    license='LGPLv3+',
    author="Elias Fekhari, Joseph Muré",
    author_email='elias.fekhari@edf.fr',
    packages=['otkerneldesign', 'test'],
    url='https://github.com/efekhari27/otkerneldesign',
    keywords=['OpenTURNS', 'KernelHerding'],
    description="Design of experiments based on kernel methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
          "numpy",
          "scipy", 
          "matplotlib",
          "openturns>=1.17"
      ],
    include_package_data=True,
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
    ],

)