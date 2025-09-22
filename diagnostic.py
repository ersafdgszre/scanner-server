import sys, subprocess
print("sys.executable:", sys.executable)
print("python version:", sys.version)
try:
    import multipart
    print("multipart import OK", getattr(multipart,'__version__','no-version'))
except Exception as e:
    print("multipart import FAILED:", repr(e))

import pkgutil
print("site-packages contains multipart?:", pkgutil.find_loader("multipart") is not None)
