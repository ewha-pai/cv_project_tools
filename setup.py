from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        name="core.correspond_pixels",
        sources=["core/correspond_pixels.pyx", "src/csa.cc", "src/Exception.cc", "src/kofn.cc", "src/match.cc", "src/Matrix.cc", "src/Random.cc", "src/String.cc", "src/Timer.cc"],
        language="c++",
        extra_compile_args=["-DNOBLAS", "-Wno-write-strings", "-Wno-format-security", "-Wno-tautological-compare", "-Wno-strict-aliasing"],
        include_dirs=["core", "/usr/include/python3.10"]
    )
]

setup(
    name="correspond_pixels",
    ext_modules=cythonize(extensions)
)