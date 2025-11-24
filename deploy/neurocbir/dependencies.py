import sys

def check_heavy_dependencies():
    for pkg_name, install_cmd in [
        ("torch", "pip install torch==2.8.0"),
        ("monai", "pip install monai==1.5.1"),
        ("nibabel", "pip install nibabel~=5.3.2"),
    ]:
        try:
            __import__(pkg_name)
        except ImportError:
            print(f"ERROR: {pkg_name} is required for NeuroCBIR.", file=sys.stderr)
            print(f"Please install it using:\n    {install_cmd}", file=sys.stderr)
            sys.exit(1)