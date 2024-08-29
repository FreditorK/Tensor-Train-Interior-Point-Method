import sys
import os
import time
from copy import copy

import numpy as np

sys.path.append(os.getcwd() + '/../')
from dataclasses import dataclass
from src.tt_ops import *
from src.tt_ops import *

x = tt_random_gaussian([5, 6, 4], shape=(2, 2))
np.set_printoptions(threshold=np.inf, linewidth=300)
print(np.round(tt_matrix_to_matrix(x), decimals=3))
x[0][:, 0, 1] = 0
x[0][:, 1, 0] = 0
print(np.round(tt_matrix_to_matrix(x), decimals=3))

