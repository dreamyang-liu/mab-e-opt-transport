from setuptools import setup

import os

BASEPATH = os.path.dirname(os.path.abspath(__file__))

setup(name='mse',
      py_modules=['mse'],
      install_requires=[
          'torch',
          'sklearn',
      ],
)