import numpy
from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "beta_binomial",
        ["beta_binomial.pyx"],
        extra_compile_args=["-fopenmp", "-O3"],
        extra_link_args=["-fopenmp"],
    )
]

setup(
    name="beta_binomial",
    ext_modules=cythonize(ext_modules, compiler_directives={"language_level": "3"}),
    include_dirs=[numpy.get_include()],
)
