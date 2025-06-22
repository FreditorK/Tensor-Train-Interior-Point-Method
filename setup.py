from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os
import scipy

# It's good practice to get the scipy include path this way
# in case the linalg headers move in the future.
scipy_include = os.path.dirname(scipy.__file__)
# You might need to go one level up for the core headers in some versions
scipy_include = os.path.join(scipy_include, '..', 'scipy')

# --- This is the key decision for performance ---
# Option 1: Generic (relies on system default being fast)
libraries_to_link = ['cblas', 'blas', 'lapack']
# Option 2: Explicitly link to OpenBLAS
# libraries_to_link = ['openblas']
# Option 3: Explicitly link to Intel MKL (often fastest on Intel)
# libraries_to_link = ['mkl_rt']


# Define common settings to avoid repetition
common_args = {
    'include_dirs': [
        np.get_include(),
        scipy_include
    ],
    'libraries': libraries_to_link,
    'extra_compile_args': ["-O3", "-march=native", "-ffast-math"],
    'extra_link_args': ["-O3", "-march=native"], # Added for good measure
    'define_macros': [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
}


extensions = [
    Extension(
        "cy_src.tt_ops_cy",
        ["cy_src/tt_ops_cy.pyx"],
        **common_args  # Use the common settings
    ),
    Extension(
        "cy_src.lgmres_cy",
        ["cy_src/lgmres_cy.pyx"],
        **common_args  # Use the common settings
    )
]

setup(
    ext_modules=cythonize(extensions),
    zip_safe=False,
    script_args=["build_ext", "--inplace"],
    install_requires=['numpy', 'scipy']
)
