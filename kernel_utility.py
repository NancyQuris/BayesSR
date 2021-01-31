import cv2
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


def weight_matrix(x, epsilon=0.001):  # weight1 and weight should be the same
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

    AI1 = np.matmul(th, KtStW0SKI)
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

    AI3 = np.matmul(th, FtKtStWSKFI)
    return AI3


def compute_Ax_h(x, W0, Ws, th, h_2d, M, N, scale_factor, eta, j, n_back, n_for, FI, Wi, ut, vt):
    Ax3 = np.zeros((M, N))
    for i in range (-1 * n_back, n_for + 1):
        if i == 0:
            Ax3_tmp = np.zeros((M, N))
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


def compute_W0(I, J0, h_2d, upscale):
    KI = frame_utility.cconv2d(h_2d, I)
    SKI = frame_utility.down_sample(KI, upscale)
    W0 = weight_matrix(SKI - J0)
    return W0


def compute_Wi(FI, J, h_2d, upscale):
    KFI = frame_utility.cconv2d(h_2d, FI)
    SKFI = frame_utility.down_sample(KFI, upscale)
    Wi = weight_matrix(SKFI - J)
    return Wi


def compute_Wk(K, I, J0, upscale):
    AK = frame_utility.cconv2d(K, I)
    SAK = frame_utility.down_sample(AK, upscale)
    Wk = weight_matrix(SAK - J0)
    return


def compute_Ws(I):
    dx = np.array([0, 0, 0, -1, 0, 1, 0, 0, 0]).reshape(3, 3)
    dy = np.array([0, -1, 0, 0, 0, 0, 0, 1, 0]).reshape(3, 3)
    DxI = frame_utility.cconv2d(dx, I)
    DyI = frame_utility.cconv2d(dy, I)
    Ws = weight_matrix(np.abs(DxI) + np.abs(DyI))
    return Ws


def compute_b1(J0, W0, theta, h_2d, scale_factor):
    W0J = np.matmul(W0, J0)
    StW0J = frame_utility.up_sample(W0J, scale_factor)
    KtStW0J = frame_utility.cconv2dt(h_2d, StW0J)
    b = np.matmul(theta, KtStW0J)
    return b


def compute_b1_k(J0, Wk, theta, I, scale_factor):
    WKJ = np.matmul(Wk, J0)
    StWkJ = frame_utility.up_sample(WKJ, scale_factor)
    AtStWkJ = frame_utility.cconv2dt(I, StWkJ)
    b = np.matmul(theta, AtStWkJ)
    return b


def compute_b3(J, Wi, theta, h_2d, upscale, ut, vt):
    WJ = np.matmul(Wi, J)
    StWJ = frame_utility.up_sample(WJ, upscale)
    KtStWJ = frame_utility.cconv2dt(h_2d, StWJ)
    FtKtStWJ = frame_utility.warped_img(KtStWJ, ut, vt)
    b3 = np.matmul(theta, FtKtStWJ)
    return b3


def compute_b_h(J, W0, theta, h_2d, scale_factor, M, N, j, n_back, n_for, Wi, ut, vt):
    b3 = np.zeros((M, N))
    for i in range(-n_back - 1, n_for):
        if i == 0:
            b3_tmp = np.zeros((M, N))
        else:
            b3_tmp = compute_b3(J[j + i], Wi[j + i], theta[j + i], h_2d, scale_factor, ut[j + i], vt[j + i])
        b3 = b3 + b3_tmp

    b1 = compute_b1(J[j], W0, theta[j], scale_factor)
    b = b1 + b3
    return b


def conj_gradient_himg(Ax, x0, b, W0, Ws, theta, h_2d, M, N, scale_factor, eta, j, n_back, n_for, FI, Wi, ut, vt, show_image):
    max_iteration = 5
    epslion = 0.1

    r = Ax - b
    p = -r
    k = 0
    x = x0
    rsize = max(r.shape)

    while k < max_iteration:
        p_m = np.reshape(p, (M, N))
        Ap_m = compute_Ax_h(p_m, W0, Ws, theta, h_2d, M, N, scale_factor, eta, j, n_back, n_for, FI, Wi, ut, vt)
        Ap = Ap_m
        alpha = np.matmul(np.transpose(r), r) / np.matmul(np.transpose(p), Ap)
        x = x + alpha * p
        r_new = r + alpha * Ap
        beta = np.matmul(np.transpose(r_new), r_new) / np.matmul(np.transpose(r), r)
        p_new = -r_new + beta * p

        diff_r_img = np.linalg.norm(r_new) / rsize
        if diff_r_img < epslion:
            break
        k += 1
        r = r_new
        p = p_new
        if show_image:
            x1 = np.reshape(x, (M, N))
            cv2.imshow('image', x1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    x = np.reshape(x, (M, N))
    return x


def conj_gradient_himg_0(Ax, x0, b, W0, Ws, th, h_2d, M, N, show_image, scale_factor, eta):
    max_iteration = 200
    epslion = 0.01
    r = Ax - b
    p = -r
    k = 0
    x = x0
    rsize = max(r.shape)

    while k < max_iteration:
        p_m = np.reshape(p, (M, N))
        Ap_m = compute_Ax(p_m, W0, Ws, th, h_2d, scale_factor, eta)
        Ap = Ap_m
        alpha = np.matmul(np.transpose(r), r) / np.matmul(np.transpose(p), Ap)
        x = x + alpha * p
        r_new = r + alpha * Ap
        beta = np.matmul(np.transpose(r_new), r_new) / np.matmul(np.transpose(r), r)
        p_new = -r_new + beta * p
        diff_r_img = np.linalg.norm(r_new) / rsize
        if diff_r_img < epslion:
            break
        k += 1
        r = r_new
        p = p_new
        if show_image:
            x1 = np.reshape(x, (M, N))
            cv2.imshow('image', x1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    x = np.reshape(x, (M, N))
    return x


def conj_gradient_kernel(Ax, x0, b, Wk, Ws, th, I, M, N, hsize, xi, scale_factor):

    max_iteration = 100
    epslion = 0.01
    # tmp = np.ones((M, N))
    # hwnd = math.floor(hsize / 2)
    # for i in range(M):
    #     for j in range(N):
    #         if i < M / 2 - hwnd or i > M / 2 + hwnd or j < N / 2 - hwnd or j > N / 2 + hwnd:
    #             tmp[i, j] = 0
    # mask = tmp
    r = Ax - b
    p = -r
    k = 0
    x = x0
    rsize = max(r.shape)

    while k < max_iteration:
        p_m = np.reshape(p, (M, N))
        Ap_m = compute_Ax_k(p_m, Wk, Ws, th, I, xi, scale_factor)
        Ap = Ap_m
        alpha = np.matmul(np.transpose(r), r) / np.matmul(np.transpose(p), Ap)
        x = x + alpha * p
        r_new = r + alpha * Ap
        beta = np.matmul(np.transpose(r_new), r_new) / np.matmul(np.transpose(r), r)
        p_new = -r_new + beta * p
        diff_r_img = np.linalg.norm(r_new) / rsize
        if diff_r_img < epslion:
            break
        k += 1
        r = r_new
        p = p_new
        x1 = np.reshape(x, (M, N))
        cv2.imshow('image', x1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    x = np.reshape(x, (M, N))
    return x