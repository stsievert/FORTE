from setuptools import setup, find_packages
try:
    from Cython.Build import cythonize
except ImportError:
    from pip import pip

    pip.main(['install', 'cython'])

    from Cython.Build import cythonize
import numpy as np

setup(
    name='FORTE',
    description='Fast Ordinal Triplet Embedding',
    url='http://github.com/lalitkumarj/FORTE',
    packages=find_packages(),
    ext_modules=cythonize(['FORTE/*.pyx',
                           'FORTE/algorithms/*.pyx',
                           'FORTE/objectives/*.pyx']),
    include_dirs=[np.get_include()]
)
