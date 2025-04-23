#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import importlib

def check_package_exists(package_name):
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

print("\nPython version: {}".format(sys.version))
print("\nPython executable: {}".format(sys.executable))
print("\nPython path:")
for path in sys.path:
    print("  - {}".format(path))

# Check audioldm package
print("\nChecking audioldm package:")
try:
    import audioldm
    print("  audioldm package found at: {}".format(audioldm.__path__[0]))
    print("  audioldm version: {}".format(getattr(audioldm, '__version__', 'Unknown')))
    print("  audioldm contains: {}".format(os.listdir(audioldm.__path__[0])))
    
    # Check specific modules
    modules_to_check = [
        'audioldm.qvim_adapter',
        'audioldm.vocaldm_utils',
        'audioldm.qvim.src.qvim_mn_baseline.ex_qvim'
    ]
    
    for module_name in modules_to_check:
        try:
            module = importlib.import_module(module_name)
            print(f"  ✓ {module_name} successfully imported")
        except ImportError as e:
            print(f"  ✗ Error importing {module_name}: {e}")
            
except ImportError as e:
    print(f"  Error importing audioldm: {e}")

# Check dependency packages
print("\nChecking dependencies:")
dependencies = [
    'torch', 'torchaudio', 'numpy', 'transformers', 'pytorch_lightning',
    'librosa', 'wandb', 'scipy'
]

for dep in dependencies:
    if check_package_exists(dep):
        print(f"  ✓ {dep} is installed")
    else:
        print(f"  ✗ {dep} is NOT installed")

# Check environment variables
print("\nRelevant environment variables:")
env_vars = ['PYTHONPATH', 'CONDA_PREFIX', 'VIRTUAL_ENV']
for var in env_vars:
    print(f"  {var}: {os.environ.get(var, 'Not set')}")

print("\nCurrent working directory: {}".format(os.getcwd()))

# Try to explicitly test the import path issue
print("\nAttempting manual package handling:")
try:
    # Add the current directory to the path if not already there
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"  Added current directory to sys.path: {current_dir}")
    
    # Try import again
    try:
        import audioldm.qvim_adapter
        print("  Successfully imported audioldm.qvim_adapter after path adjustment")
    except ImportError as e:
        print(f"  Still unable to import audioldm.qvim_adapter: {e}")
        
    # Check if the file exists on disk
    expected_path = os.path.join(current_dir, 'audioldm', 'qvim_adapter.py')
    if os.path.exists(expected_path):
        print(f"  File exists at: {expected_path}")
    else:
        print(f"  File does NOT exist at: {expected_path}")
        
except Exception as e:
    print(f"  Error during manual package handling: {e}")