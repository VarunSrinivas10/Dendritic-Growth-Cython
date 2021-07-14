from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
  name='func_body',
  ext_modules=[Extension('func_body', ['func_body.pyx'],include_dirs=[numpy.get_include()]),],
  cmdclass={'build_ext': build_ext},
)

