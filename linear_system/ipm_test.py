import sys
import os

sys.path.append(os.getcwd() + '/../')

import numpy as np
import scipy.linalg as lin
import copy
import time
from typing import List
from src.tt_ops import *
import scikit_tt as scitt
from scikit_tt.solvers.sle import als as sota_als
from src.tt_ops import *

tt_ipm()