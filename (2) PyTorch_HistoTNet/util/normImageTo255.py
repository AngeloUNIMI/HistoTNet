import numpy as np

def normImageTo255(x):
    minX = np.min(x)
    maxX = np.max(x)
    x2 = x - minX
    x3 = x2 / (maxX-minX)
    x4 = x3 * 255
    return x4

