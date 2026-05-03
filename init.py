"""
Exam startup — run with: ipython -i init.py
"""

# --- Suppress Out[] display (prints still show) ---
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "none"

# --- Core libraries ---
import numpy as np
from scipy import stats

# --- Toolkit ---
from compute import *
from plots import *
from stats_compute import *
from stats_plots import *

print("Ready.")
