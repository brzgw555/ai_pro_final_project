import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = '0.0.1'


sources = [
    'source/pybind.cpp',
    'source/Tensor.cu',
    'source/Activate.cu',
    'source/Linear.cu',
    'source/Pool.cu',
    'source/Softmax.cu',
    'source/CrossEntropyLoss.cu',
    'source/Conv.cu'
]


extra_compile_args = {
    'cxx': [
        '/std:c++17', 
        '/O2'          
    ],
    'nvcc': [
        '-std=c++17', 
        '-O3',         
        '-arch=sm_89'  
    ]
}


extra_link_args = []

if sys.platform == 'win32':
    
    extra_link_args.append('/MACHINE:x64')
    
    
    # python_lib_path = r'C:\Users\TrailblazerWu\anaconda3\envs\resvid\libs'
    # extra_link_args.append(f'/LIBPATH:{python_lib_path}')

setup(
    name='mytensor',
    version=__version__,
    author='your_name',
    author_email='your_name@pku.edu.cn',
    packages=find_packages(),
    zip_safe=False,
    install_requires=['torch'],
    python_requires='>=3.8',
    license='MIT',
    ext_modules=[
        CUDAExtension(
            name='mytensor',  
            sources=sources,
            
            include_dirs=[],
            extra_compile_args=extra_compile_args,  
            extra_link_args=extra_link_args        
        )
    ],
    cmdclass={
        'build_ext': BuildExtension 
    },
    classifiers=[
        'License :: OSI Approved :: MIT License',
    ],
)