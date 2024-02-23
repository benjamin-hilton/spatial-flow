from distutils.core import setup, Extension
import numpy as np

module1 = Extension('radiationabm',
                    sources=['radiationabmmodule.cpp'],
                    extra_compile_args=['-std=c++11', '-O2', '-I/rds/general/user/bah15/home/anaconda3/include/python3.6m',
                                        '-I/rds/general/user/bah15/home/anaconda3/lib/python3.6/site-packages/numpy/core/include'])

setup(name='RadiationABMModule',
      version='0.1',
      description='Implements a radiation model ABM with interactions.',
      include_dirs=[np.get_include()],
      ext_modules=[module1])
