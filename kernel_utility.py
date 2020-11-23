import numpy as np
import math


def fspecial_gaussian(size, sigma):
    """
        Function to mimic the 'fspecial' gaussian MATLAB function
        from https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    """

    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def create_h(sigma_h, h_size, mode, width, height):
    if sigma_h > width or sigma_h > height:
        print('Blur kernel must be smaller than image')
        exit(1)
    x = np.linspace(-h_size / 2, h_size / 2, h_size)
    x = (-x ** 2) / (2 * math.pow(sigma_h, 2))
    h_1d = np.exp(x)
    h_1d = h_1d / h_1d.sum()
    if mode == 1:
        h_2d = np.kron(np.transpose(h_1d), h_1d)
        return h_1d, h_2d
    elif mode == 2:
        h_2d = np.ones((h_size, h_size)) / (h_size * h_size)
        return h_1d, h_2d

