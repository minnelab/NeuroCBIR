from setuptools import setup, find_packages
import os


# Determine path relative to this setup.py
HERE = os.path.abspath(os.path.dirname(__file__))
version_file = os.path.join(HERE, "neurocbir", "version.py")

# Version
version_file = os.path.join(HERE, "version.py")
version_ns = {}
with open(version_file, "r") as f:
    exec(f.read(), {}, version_ns)
package_version = version_ns["__version__"]

# Read the long description from README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Core dependencies (lightweight)
install_requires = [
    "numpy~=2.2.6",
    "pandas~=2.3.3",
    "fastparquet~=2024.11.0",
    "pyarrow~=21.0.0",
    "scipy~=1.15.3",
    "scikit-learn~=1.7.2",
    "tabulate~=0.9.0",
    "PyYAML~=6.0.3",
]

# Optional / heavy dependencies
extras_require = {
    "full": [
        "torch==2.8.0",       # User must install manually with PyTorch index for CPU/GPU
        "monai==1.5.1",
        "nibabel~=5.3.2",
    ]
}



setup(
    name="neurocbir",
    version=package_version,
    author='Félix Nieto-del-Amor',
    author_email='fenda@kth.se',
    description="NeuroCBIR: A Public Content-Based Image Retrieval System for Whole-Brain and Region-Specific MRI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/feniede/NeuroCBIR",
    package_dir={'': '.'},
    packages=find_packages(where='.'),
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
