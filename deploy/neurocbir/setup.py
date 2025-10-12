import os
from setuptools import setup, find_packages

def read_requirements(file_path):
    """Reads a requirements file and returns a list of dependencies."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# The requirements file is in the same directory as setup.py
requirements_path = 'requirements.txt'
install_requires = read_requirements(requirements_path)

# The README is two levels up from this file's location
readme_path = 'README.md'
with open(readme_path, 'r') as f:
    long_description = f.read()

setup(
    name='neurocbir',
    version='0.1.0',
    author='Félix Nieto-del-Amor',
    author_email='fenda@kth.se',
    description='A Public Content-Based Image Retrieval System for Whole-Brain and Region-Specific MRI',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/feniede/NeuroCBIR',
    # Since setup.py is inside the package source, the package root is the current directory
    package_dir={'': '.'},
    packages=find_packages(where='.'),
    install_requires=install_requires,
    include_package_data=True,
    package_data={
        'neurocbir': [
            'configs/*.yaml',
        ],
    },
    entry_points={
        'console_scripts': [
            'neurocbir=neurocbir.main:main',
        ],
    },
    python_requires='>=3.10',
)