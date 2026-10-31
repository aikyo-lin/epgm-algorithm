import os
import sys

# Ensure epgm-algorithm directory is in sys.path when running pytest from any directory
ALGO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ALGO_DIR not in sys.path:
    sys.path.insert(0, ALGO_DIR)
