#!/usr/bin/env python
from setuptools import setup, find_packages

print(find_packages)
if __name__ == "__main__":
    setup(
    name="makedataset",
    version="0.0",
    packages=find_packages(exclude=['cohorts']),
    install_requires=["numpy", "nibabel", "pandas", "tqdm"])
