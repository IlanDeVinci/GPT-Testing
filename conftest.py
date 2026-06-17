import os
import sys
from pathlib import Path

# Evite une access violation Windows quand plusieurs modules importent torch
# (conflit de runtime OpenMP/MKL charge en double).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).parent))
