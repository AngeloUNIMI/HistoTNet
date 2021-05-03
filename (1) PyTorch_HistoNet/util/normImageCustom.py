import numpy as np

def normImageCustom(x, minX, maxX):
    x2 = x - minX
    x3 = x2 / maxX
    return x3

