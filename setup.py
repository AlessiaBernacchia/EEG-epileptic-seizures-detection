# setup.py makes this repository installable as a Python package.
# This is useful for notebooks and scripts because it allows imports
# like `from utils.collection.collect import ...` without manually
# modifying sys.path.

from setuptools import setup, find_packages

setup(
    name="eeg_seizure_detection",
    version="0.1.0",
    description="EEG seizure detection project package",
    packages=find_packages(),
    python_requires=">=3.8",
)
