import numpy as np
import math
import frame_utility


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


def weight_matrix(x, epsilon):
    m, n = x.shape
    W = np.zeros(m, n)

    for i in range(m):
        for j in range(n):
            W[i, j] = 1 / math.sqrt(x[i, j] ** 2 + epsilon ** 2)
    return W

def compute_Ax1_k(K, I, Wk, th, scale_factor):

    AK = frame_utility.cconv2d(K, I)
    SAK = frame_utility.down_sample(AK, scale_factor)
    WkSAK = np.matmul(Wk, SAK)
    StWkSAK = frame_utility.up_sample(WkSAK, scale_factor)
    AtStWkSAK = frame_utility.cconv2dt(I, StWkSAK)

    AK1 = th * AtStWkSAK

    return AK1


def compute_Ax1(I, W0, th, h_2d, scale_factor):
    KI = frame_utility.cconv2d(h_2d, I)
    SKI = frame_utility.down_sample(KI, scale_factor)
    W0SKI = np.matmul(W0, SKI)
    StW0SKI = frame_utility.up_sample(W0SKI, scale_factor)
    KtStW0SKI = frame_utility.cconv2dt(h_2d, StW0SKI)

    AI1 = th * KtStW0SKI
    return AI1


def compute_Ax2(I, Ws):
    dx = np.array([0, 0, 0, -1, 0, 1, 0, 0, 0]).reshape(3, 3)
    dy = np.array([0, -1, 0, 0, 0, 0, 0, 1, 0]).reshape(3, 3)

    DxI = frame_utility.cconv2d(dx, I)
    WsDxI = np.matmul(Ws, DxI)
    DxtWsDxI = frame_utility.cconv2dt(dx, WsDxI)
    DyI = frame_utility.cconv2d(dy, I)
    WsDyI = np.matmul(Ws, DyI)
    DytWsDyI = frame_utility.cconv2dt(dy, WsDyI)

    AI2 = DxtWsDxI + DytWsDyI
    return AI2


def compute_Ax3(FI, Wi, th, h_2d, scale_factor, ut, vt):
    KFI = frame_utility.cconv2d(h_2d, FI)
    SKFI = frame_utility.down_sample(KFI, scale_factor)
    WSKFI = np.matmul(Wi, SKFI)
    StWSKFI = frame_utility.up_sample(WSKFI, scale_factor)
    KtStWSKFI = frame_utility.cconv2dt(h_2d, StWSKFI)
    FtKtStWSKFI = frame_utility.warped_img(KtStWSKFI, ut, vt)

    AI3 = th * FtKtStWSKFI
    return AI3


def compute_Ax_h(x, W0, Ws, th, h_2d, height, width, scale_factor, eta, j, n_back, n_for, FI, Wi, ut, vt):
    Ax3 = np.zeros((height, width))
    for i in range (-1 * n_back, n_for + 1):
        if i == 0:
            Ax3_tmp = np.zeros((height, width))
        else:
            Ax3_tmp = compute_Ax3(FI[j + 1], Wi[j+i], th[j+i], h_2d, scale_factor, ut[j+i], vt[j+i])
        Ax3 = Ax3 + Ax3_tmp

    Ax1 = compute_Ax1(x, W0, th[j], h_2d, scale_factor)
    Ax2 = compute_Ax2(x,Ws)
    Ax = Ax1 + eta * Ax2 + Ax3
    return Ax


def compute_Ax_k(x, Wk, Ws, th, I, xi, scale_factor):
    Ax1 = compute_Ax1_k(x,I,Wk,th, scale_factor)
    Ax2 = compute_Ax2(x, Ws)
    Ax = Ax1 + xi * Ax2

    return Ax


def compute_Ax(I, W0, Ws, th, h_2d, scale_factor, eta):
    AI1 = compute_Ax1(I, W0, th, h_2d, scale_factor)
    AI2 = compute_Ax2(I, Ws)
    Ax = AI1 + eta * AI2
    return Ax
