#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from distutils.core import setup, Extension
import numpy as np

ext_modules = [Extension('_cgeo', sources=['_cgeo.cpp'])]

setup(name='_cgeo', version='1.0', include_dirs=[np.get_include()],
      ext_modules=ext_modules)
