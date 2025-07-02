import sys
import site
from pathlib import Path

arbeit_path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(arbeit_path))

print("=== Loaded Modules at Startup ===")
for k in sys.modules.keys():
    print(k)
print("==================================")

import builtins
import traceback

_real_import = builtins.__import__

def verbose_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "mpi4py.MPI" or name.startswith("mpi4py.MPI"):
        print("\n⚠️ MPI4PY IMPORT DETECTED ⚠️")
        traceback.print_stack()
        print("\n")
    return _real_import(name, globals, locals, fromlist, level)

builtins.__import__ = verbose_import

# Now import your module
import Learning.Training_loops as node
from Validation.validation_funcs import *
from amuse.lab import Particle, Particles