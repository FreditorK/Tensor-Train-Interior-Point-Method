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
    'libraries': ['openblas'],
    'extra_compile_args': [
        "-O3", "-march=native", "-flto=16", "-fopenmp",
        "-funroll-loops", "-ftree-vectorize", "-fprefetch-loop-arrays",
        "-fno-math-errno", "-fno-semantic-interposition"
    ],
    'extra_link_args': [
        "-O3", "-march=native", "-flto=16", "-fopenmp", "-Wl,-O1", "-Wl,--as-needed"
    ],
    'define_macros': [
        ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
        ('CYTHON_FAST_PYCCALL', '1'),
        ('CYTHON_USE_TYPE_SLOTS', '1'),
        ('CYTHON_USE_PYTYPE_LOOKUP', '1'),
        ('CYTHON_USE_DICT_VERSIONS', '1')
    ]
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
    script_args=["build_ext", "--inplace"],
    install_requires=['numpy', 'scipy']
)