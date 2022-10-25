from setuptools import setup
from distutils.extension import Extension
from Cython.Build import cythonize


setup(ext_modules = cythonize(
           "c_fast_model.pyx",                 # our Cython source
      ))
