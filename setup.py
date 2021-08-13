from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='postcode',
    version='0.1',
    packages=find_packages(where='source-code'),
    package_dir={'': 'source-code'},
    py_modules=[splitext(basename(path))[0] for path in glob('source-code/*.py')],
)