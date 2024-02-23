from distutils.core import setup, Extension
import numpy as np

module1 = Extension('spatialmodels',
                    sources=['spatialmodelsmodule.cpp'],
                    extra_compile_args=['-std=c++17', '-O2'])

setup(name='SpatialModelsModule',
      version='0.1',
      description='Implements radiation and gravity models.',
      include_dirs=[np.get_include()],
      ext_modules=[module1])
