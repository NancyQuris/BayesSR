import cv2
import numpy as np
import math
from scipy import interpolate

def frame_cutting(path):
    video_capture = cv2.VideoCapture(path)
    success, image = video_capture.read()
    count = 0
    while success:
        cv2.imwrite(path+"/original_frame%d.png" % count, image)
        success, image = video_capture.read()
        count += 1
    video_capture.release()


def resize(image, scale_factor, interpolation_method, is_image_array=False):
    scale_percent = scale_factor * 100

    if is_image_array:
        result = []
        for i in image:
            width = int(i.shape[1] * scale_percent / 100)
            height = int(i.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(i, dim, interpolation=interpolation_method)
            result.append(resized)
        return result
    else:
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(image, dim, interpolation=interpolation_method)
        return resized


def load_frame(path, start_frame_index, num_of_frame, scale_factor, interpolation_method=cv2.INTER_CUBIC):
    original_images = []
    scaled_images = []
    for i in range(start_frame_index, start_frame_index + num_of_frame):
        original_frame = cv2.imread(path+"/original_frame%d.png" % i)
        original_images.append(original_frame)
        resized = resize(original_frame, scale_factor, interpolation_method)
        scaled_images.append(resized)
    return original_images, scaled_images


def bgr2ycrcb(frames):
    converted_frames = []
    for f in frames:
        converted_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb))
    return converted_frames


def split_ycrcb_frames(frames):
    y = []
    cr = []
    cb = []
    for f in frames:
        current_y, current_cr, current_cb = cv2.split(f)
        y.append(current_y)
        cr.append(current_cr)
        cb.append(current_cb)
    return y, cr, cb


def add_noise(ycbcr_l, mean, variance):
    for f in ycbcr_l:
        gaussian_noise = np.random.normal(mean, math.sqrt(variance), f.shape)
        ycbcr_l[i] = f + gaussian_noise

    return ycbcr_l


def zero_pad(image, shape, position='corner'): # adapted from https://github.com/aboucaud/pypher/blob/master/pypher/pypher.py
    """
    Extends image to a certain size with zeros
    Parameters
    ----------
    image: real 2d `numpy.ndarray`
        Input image
    shape: tuple of int
        Desired output shape of the image
    position : str, optional
        The position of the input image in the output one:
            * 'corner'
                top-left corner (default)
            * 'center'
                centered
    Returns
    -------
    padded_img: real `numpy.ndarray`
        The zero-padded image
    """
    shape = np.asarray(shape, dtype=int)
    imshape = np.asarray(image.shape, dtype=int)

    if np.alltrue(imshape == shape):
        return image

    if np.any(shape <= 0):
        raise ValueError("ZERO_PAD: null or negative shape given")

    dshape = shape - imshape
    if np.any(dshape < 0):
        raise ValueError("ZERO_PAD: target size smaller than source one")

    pad_img = np.zeros(shape, dtype=image.dtype)

    idx, idy = np.indices(imshape)

    if position == 'center':
        if np.any(dshape % 2 != 0):
            raise ValueError("ZERO_PAD: source and target shapes "
                             "have different parity.")
        offx, offy = dshape // 2
    else:
        offx, offy = (0, 0)

    pad_img[idx + offx, idy + offy] = image

    return pad_img


def psf2otf(psf, shape):  # adapted from https://github.com/aboucaud/pypher/blob/master/pypher/pypher.py
    """
    Convert point-spread function to optical transfer function.
    Compute the Fast Fourier Transform (FFT) of the point-spread
    function (PSF) array and creates the optical transfer function (OTF)
    array that is not influenced by the PSF off-centering.
    By default, the OTF array is the same size as the PSF array.
    To ensure that the OTF is not altered due to PSF off-centering, PSF2OTF
    post-pads the PSF array (down or to the right) with zeros to match
    dimensions specified in OUTSIZE, then circularly shifts the values of
    the PSF array up (or to the left) until the central pixel reaches (1,1)
    position.
    Parameters
    ----------
    psf : `numpy.ndarray`
        PSF array
    shape : int
        Output shape of the OTF array
    Returns
    -------
    otf : `numpy.ndarray`
        OTF array
    Notes
    -----
    Adapted from MATLAB psf2otf function
    """
    if np.all(psf == 0):
        return np.zeros_like(psf)

    inshape = psf.shape
    # Pad the PSF to outsize
    psf = zero_pad(psf, shape, position='corner')

    # Circularly shift OTF so that the 'center' of the PSF is
    # [0,0] element of the array
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, -int(axis_size / 2), axis=axis)

    # Compute the OTF
    otf = np.fft.fft2(psf)

    # Estimate the rough number of operations involved in the FFT
    # and discard the PSF imaginary part if within roundoff error
    # roundoff error  = machine epsilon = sys.float_info.epsilon
    # or np.finfo().eps
    n_ops = np.sum(psf.size * np.log2(psf.shape))
    otf = np.real_if_close(otf, tol=n_ops)

    return otf


def cconv2d(h, x):
    m, n = x.shape
    mh, nh = h.shape

    if m < mh or n < nh:
        print("size of kernel must be bigger than image")
    else:
        fft_h = psf2otf(h, x.shape)
        fft_x = np.fft.fft2(x)
        fft_y = fft_h * fft_x

        return np.fft.ifft2(fft_y)


def cconv2d_k(h, x):
    m, n = x.shape
    mh, nh = h.shape

    if m < mh or n < nh:
        print("size of kernel must be bigger than image")
    else:
        fft_y = np.conj(psf2otf(x, x.shape)) * np.fft.fft2(h)
        return np.fft.ifft2(fft_y)


def cconv2dt(h, x):
    m, n = x.shape
    mh, nh = h.shape

    if m < mh or n < nh:
        print("size of kernel must be bigger than image")
    else:
        fft_h = np.conj(psf2otf(h, x.shape))
        fft_x = np.fft.fft2(x)
        fft_y = fft_h * fft_x
        return np.fft.ifft2(fft_y)


def down_sample(x, scale):
    m, n = x.shape
    if m % scale != 0 or n % scale != 0:
        print("size of x must be divided by scale")
    else:
        y = np.zeros(m / scale, n / scale)
        for i in range(m/scale):
            for j in range(n/scale):
                y[i, j] = x[i * scale, j * scale]
        return y


def up_sample(x, scale):
    m, n = x.shape
    y = np.zeros(m * scale, n * scale)
    for i in range(m):
        for j in range(n):
            y[i * scale, j * scale] = x[i, j]
    return y


def warped_img(I, u, v):
    m, n = I.shape
    x_grid = np.arange(n)
    y_grid = np.arange(m)
    xPosv, yPosv = np.meshgrid(x_grid, y_grid)

    xPosv = xPosv - u
    yPosv = yPosv - v

    for x in xPosv:
        if x <= 1:
            x = 1
        elif x >= n:
            x = n

    for y in yPosv:
        if y <= 1:
            y = 1
        elif y >= m:
            y = m

    FI = interpolate.interp2d(I, xPosv, yPosv, kind='cubic')

    return FI

def create_low_res_with_S_K(high_res_frames, h_2d, scale_factor, num_of_frames):
    low_res_frames = []
    for i in range(num_of_frames):
        current_original_frame = high_res_frames[i]
        b, g, r = cv2.split(current_original_frame)
        blurred_b = cconv2d(h_2d, b)
        blurred_g = cconv2d(h_2d, g)
        blurred_r = cconv2d(h_2d, r)

        downsample_b = down_sample(blurred_b, scale_factor)
        downsample_g = down_sample(blurred_g, scale_factor)
        downsample_r = down_sample(blurred_r, scale_factor)

        low_res_frames.append(cv2.merge((downsample_b, downsample_g, downsample_r)))

    return low_res_frames


def create_initial_I(frames, scale_factor):
    result = []
    for f in frames:
        if scale_factor == 2:
            h = np.asarray([0.25, 0.5, 0.25, 0.5, 1, 0.5, 0.25, 0.5, 0.25]).reshape((3, 3))
            tmp = up_sample(f, scale_factor)
            result.append(cconv2d(h, tmp))
        else:
            result.append(resize(f, scale_factor, cv2.INTER_LINEAR))
    return result
