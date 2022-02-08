from setuptools import setup
from torch.utils import cpp_extension

setup(
    name='static_nms_cpp',
    ext_modules=[
        # cpp_extension.CppExtension(
        #     'static_nms', ['static_nms.cpp']
        # ),
        cpp_extension.CppExtension(
            'static_batched_nms', ['static_nms.cpp']
        )
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)


