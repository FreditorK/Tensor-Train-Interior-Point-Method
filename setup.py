from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
from multiprocessing import cpu_count
import scipy
import os

scipy_include = os.path.dirname(scipy.__file__)
scipy_include = os.path.join(scipy_include, '..', 'scipy')

common_args = {
    'include_dirs': [np.get_include(), scipy_include],
    'libraries': ['mkl_rt'],  # Use MKL runtime dispatcher
    'extra_compile_args': [
        "-O3", "-march=native", "-ffast-math", "-funroll-loops", "-fopenmp"
    ],
    'extra_link_args': [
        "-lmkl_rt", "-fopenmp"
    ],
    'define_macros': [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
}

extensions = [
    Extension("cy_src.tt_ops_cy", ["cy_src/tt_ops_cy.pyx"], **common_args),
    Extension("cy_src.lgmres_cy", ["cy_src/lgmres_cy.pyx"], **common_args)
]

setup(
    ext_modules=cythonize(
        extensions,
        nthreads=cpu_count(),
        language_level=3
    ),
    zip_safe=False,
    script_args=["build_ext", "--inplace", "--force"],
    install_requires=['numpy', 'scipy']
)
