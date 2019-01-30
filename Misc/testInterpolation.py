import numpy as np
from scipy import interpolate
import scipy

x = np.arange(8)
y = [1, 5, 7, 11, 17, 26, 33, 45]

z = interpolate.interp1d(x, y)
print()

