# setup.py
from setuptools import setup, find_packages

setup(
    name="netisg",
    version="12.1.0", # Version updated
    description="NETISG-XII: A Universal and Robust Deep Learning Skeletonization Framework",
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'scipy',
        'opencv-python',
        'rembg',
        'onnxruntime',
        'tqdm',
        'open3d',
        'torch',
        'Pillow' # Added PIL for safety
    ]
)
