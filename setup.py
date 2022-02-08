#!/usr/bin/env python

import os

import torch
from setuptools import find_packages
from setuptools import setup
import distutils.command.build

from torch.utils.cpp_extension import CppExtension

requirements = ["torch", "torchvision"]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.join(this_dir, 'lib')
    if not os.path.isdir(lib_dir):
        os.mkdir(lib_dir)
    extensions_dir = os.path.join(this_dir, 'src')

    extension = CppExtension

    custom_ops_sources = [os.path.join(extensions_dir, 'vision.cpp')]
    custom_ops_sources += [os.path.join(extensions_dir, 'cpu', 'nms_cpu.cpp')]
    custom_ops_sources += [os.path.join(extensions_dir, 'cpu', 'static_nms.cpp')]

    extra_compile_args = {'cxx': []}
    define_macros = []

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "lib.torch_ext",
            custom_ops_sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        ),
    ]

    return ext_modules

setup(
    name="torch_ext_ops",
    version="0.1",
    author="Intellif",
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension,
              "build": distutils.command.build.build}
)
