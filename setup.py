from distutils.core import setup
from Cython.Build import cythonize
setup(
    ext_modules = cythonize("src/tt_ops_cy.pyx")
)