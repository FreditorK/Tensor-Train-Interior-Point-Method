from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "cy_src.tt_ops_cy",  # Name of the Cython module
        ["cy_src/tt_ops_cy.pyx"],  # Source file
        include_dirs=[np.get_include()],  # This line ensures that NumPy headers are included
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    )
]

setup(
    ext_modules=cythonize(extensions),
    zip_safe=False,
    script_args=["build_ext", "--inplace"],
)