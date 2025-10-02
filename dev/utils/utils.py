import importlib.util
import os
import sys

def load_config_from_path(path):
    spec = importlib.util.spec_from_file_location("config_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if hasattr(module, "config"):
        return module.config
    else:
        raise AttributeError(f"No 'config' variable found in {path}")