import cv2
import numpy as np
import math
from scipy import ndimage, interpolate


def get_dir_name(path):
    path_length = len(path)
    position_of_last_slash = 0
    for i in range(path_length - 1, -1, -1):
        current_char = path[i]
        if current_char == '/':
            position_of_last_slash = i + 1
            break
    return path[:position_of_last_slash]


def frame_cutting(path):
    video_capture = cv2.VideoCapture(path)
    success, image = video_capture.read()
    count = 0
    dir_path = get_dir_name(path)
    while success:
        cv2.imwrite(dir_path + "original_frame%d.png" % count, image)
        success, image = video_capture.read()
        count += 1
    video_capture.release()


def resize(image, scale_factor, interpolation_method, is_image_array=False):
    # if image is nparray, add .astype('uint8') to convert datatype
    if is_image_array:
        result = []
        for i in image:
            dim = (int(i.shape[1] * scale_factor), int(i.shape[0] * scale_factor))
            resized = cv2.resize(i, dim, interpolation=interpolation_method)
            result.append(resized)
        return result
    else:
        dim = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
        resized = cv2.resize(image, dim, interpolation=interpolation_method)
        return resized


def load_frame(path, start_frame_index, num_of_frame, scale_factor, interpolation_method=cv2.INTER_CUBIC):
    original_images = []
    scaled_images = []
    dir_path = get_dir_name(path)
    for i in range(start_frame_index, start_frame_index + num_of_frame):
        original_frame = cv2.imread(dir_path + "original_frame%d.png" % i)
        original_images.append(original_frame)
        resized = resize(original_frame, scale_factor, interpolation_method)
        scaled_images.append(resized)
    return original_images, scaled_images


def bgr2ycrcb(frames):
    converted_frames = []
    for f in frames:
        converted_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2YCrCb))
    return converted_frames


def y2rgb(frames, ycrcb_frames):
    converted_frames = []
    for i in range(len(frames)):
        current_ycrcb_frame = ycrcb_frames[i]
        current_ycrcb_frame[:, :, 0] = frames[i]
        converted_frames.append(cv2.cvtColor(np.float32(current_ycrcb_frame), cv2.COLOR_YCrCb2RGB))
    return converted_frames


def pad_zeros(image, pad_height, pad_width):
    height, width = image.shape
    new_height, new_width = height + pad_height * 2, width + pad_width * 2
    padded_image = np.zeros((new_height, new_width))
    padded_image[pad_height:new_height - pad_height, pad_width:new_width - pad_width] = image
    return padded_image


def cconv2d(kernel, image):
    # print(kernel.shape)
    # print(image.shape)
    # Hi, Wi = image.shape
    # Hk, Wk = kernel.shape
    # flipped_kernel = np.flip(np.flip(kernel, 0), 1)
    # reshaped_kernel = flipped_kernel.reshape([Hk * Wk, 1])
    # padded_image = pad_zeros(image, Hk // 2, Wk // 2)
    # filter_window = np.zeros((Hi * Wi, Hk * Wk))
    # image_x = 0
    # image_y = 0
    # for i in range(Hi * Wi):
    #     y_range = image_y + Hk
    #     x_range = image_x + Wk
    #     window = padded_image[image_y: y_range, image_x: x_range]
    #     filter_window[i] = window.reshape([1, Hk * Wk])
    #     if image_x == Wi - 1:
    #         image_x = 0
    #         image_y += 1
    #     else:
    #         image_x += 1
    # result = np.matmul(filter_window, reshaped_kernel)
    # filtered_image = result.reshape([Hi, Wi])
    # return filtered_image
    #
    # result = signal.convolve2d(image, kernel, mode='same')
    # return result

    # https://stackoverflow.com/questions/5710842/fastest-2d-convolution-or-image-filter-in-python
    return np.fft.irfft2(np.fft.rfft2(image) * np.fft.rfft2(kernel, image.shape))


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


def add_noise(ycbcr_l, mean, std):
    result = []
    for f in ycbcr_l:
        gaussian_noise = np.random.normal(mean, std, f.shape)
        result.append(f + gaussian_noise)
    return result


def down_sample(f, ratio):
    width, height = f.shape[1], f.shape[0]
    return f[0:height:ratio, 0:width:ratio]


def up_sample(f, ratio):
    width, height = f.shape[1], f.shape[0]
    new_width, new_height = width * ratio, height * ratio
    res_image = np.zeros((new_height, new_width))
    res_image[0:new_height:ratio, 0:new_width:ratio] = f
    return res_image


def psf2otf(psf, shape):
    # https://blog.csdn.net/weixin_43890288/article/details/105676416
    psf_size = np.array(psf.shape)
    shape = np.array(shape)
    pad_size = shape - psf_size
    psf = np.pad(psf, ((0, pad_size[0]), (0, pad_size[1])), 'constant')
    for i in range(len(psf_size)):
        psf = np.roll(psf, -int(psf_size[i] / 2), i)
    otf = np.fft.fftn(psf)
    n_elem = np.prod(psf_size)
    n_ops = 0
    for k in range(len(psf_size)):
        nffts = n_elem / psf_size[k]
        n_ops = n_ops + psf_size[k] * np.log2(psf_size[k]) * nffts
    if np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf)) <= n_ops * np.finfo(np.float32).eps:
        otf = np.real(otf)
    return otf


def otf2psf(otf, shape):
    # https://blog.csdn.net/weixin_42206235/article/details/94037686
    planes = [np.zeros(otf.shape), np.zeros(otf.shape)]
    psf_circle = np.zeros(otf.shape)
    cv2.dft(np.float32(otf), np.float32(psf_circle), cv2.DFT_INVERSE + cv2.DFT_SCALE, 0)
    cv2.split(psf_circle, planes)

    psf = planes[0].copy()

    x = psf.shape[0]
    y = psf.shape[1]
    cx = int((otf.shape[0] + 1) / 2)
    cy = int((otf.shape[1] + 1) / 2)

    p0 = planes[0][0:cx, 0:cy].copy()
    p1 = planes[0][cx:x, 0:cy].copy()
    p2 = planes[0][0:cx, cy:y].copy()
    p3 = planes[0][cx:x, cy:y].copy()

    psf[x - cx:x, y - cy:y] = p0.copy()
    psf[0:x - cx, 0:y - cy] = p3.copy()
    psf[0:x - cx, y - cy:y] = p1.copy()
    psf[x - cx:x, 0:y - cy] = p2.copy()
    return psf[0:shape[0], 0:shape[1]]


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


def change_scale(x, scale):
    new_height = int(x.shape[0] * scale)
    new_width = int(x.shape[1] * scale)
    new_image = np.zeros((new_height, new_width, 3), dtype='uint8') # if YCrCb or RGB
    if len(x.shape) == 2:
        new_image = np.zeros((new_height, new_width), dtype='uint8')
    original_height = len(x)
    original_width = len(x[0])
    for h in range(new_height):
        for w in range(new_width):
            h_index = int(original_height / new_height * h)
            w_index = int(original_width / new_width * w)
            new_image[h][w] = x[h_index][w_index]
    return new_image


def warped_img(I, u, v):
    m, n = I.shape
    x_grid = np.arange(n)
    y_grid = np.arange(m)

    xPosv, yPosv = np.meshgrid(x_grid, y_grid)
    xPosv = xPosv - u
    yPosv = yPosv - v

    xPosv[xPosv <= 1] = 1
    xPosv[xPosv >= n] = n
    yPosv[yPosv <= 1] = 1
    yPosv[yPosv >= m] = m

    FI = ndimage.map_coordinates(I, [xPosv.ravel(), yPosv.ravel()], order=3, mode='nearest').reshape(I.shape)
    return FI


def create_low_res_with_S_K(high_res_frames, h_2d, scale_factor, num_of_frames):
    low_res_frames = []
    for i in range(num_of_frames):
        current_original_frame = high_res_frames[i]
        b, g, r = cv2.split(current_original_frame)
        blurred_b = cconv2d(h_2d, b)
        blurred_g = cconv2d(h_2d, g)
        blurred_r = cconv2d(h_2d, r)

        downsample_b = change_scale(blurred_b, scale_factor)
        downsample_g = change_scale(blurred_g, scale_factor)
        downsample_r = change_scale(blurred_r, scale_factor)

        low_res_frames.append(cv2.merge((downsample_b, downsample_g, downsample_r)))

    return low_res_frames


def create_initial_I(frames, scale_factor):
    result = []
    for f in frames:
        if scale_factor == 2:
            h = np.asarray([0.25, 0.5, 0.25, 0.5, 1, 0.5, 0.25, 0.5, 0.25]).reshape((3, 3))
            tmp = change_scale(f, scale_factor)
            result.append(cconv2d(h, tmp))
        else:
            result.append(resize(f, scale_factor, cv2.INTER_LINEAR))
    return result


def shift_adjust(img, upscale, direction):
    N = len(img)
    img_sh = img
    if upscale == 2:
        shift = 0.5
    elif upscale == 3:
        shift = 1
    elif upscale == 4:
        shift = 1.5
    else:
        shift = 0

    if img[0].shape[2] == 3:
        img0 = img[0]
        m, n = img0[:, :, 0].shape
    else:
        m, n = img[0].shape
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, m))

    if direction == 1:
        x1 = x + shift
        y1 = y + shift
    else:
        x1 = x - shift
        y1 = y - shift

    x1[x1 < 1] = 1
    x1[x1 >= n] = n
    y1[y1 < 1] = 1
    y1[y1 >= m] = m

    if img[0].shape[2] == 3:
        for j in range(N):
            for i in range(3):
                # img_sh[j][:, :, i] = ndimage.map_coordinates(x, y, img[j][:, :, i], x1, y1)
                # F = interpolate.RectBivariateSpline(x, y, img[j][:, :, i])
                # img_sh[j][:, :, i] = np.diagonal(F(x1, y1)).reshape((m, n))
                img_sh[j][:, :, i] = ndimage.map_coordinates(img[j][:, :, i], [x1.ravel(), y1.ravel()],
                                                             order=3, mode='nearest').reshape((m, n))
    else:
        for j in range(N):
            # img_sh[j][:, :] = ndimage.map_coordinates(x, y, img_sh[j][:, :], x1, y1)
            # F = interpolate.RectBivariateSpline(x, y, img[j][:, :])
            # img_sh[j][:, :] = np.diagonal(F(x1, y1)).reshape((m, n))
            img_sh[j][:, :] = ndimage.map_coordinates(img[j][:, :], [x1.ravel(), y1.ravel()],
                                                         order=3, mode='nearest').reshape((m, n))
    return img_sh


def get_RMSE(I1, I2):
    diff = I1 - I2
    m = np.sum(diff ** 2) / diff.size
    return math.sqrt(m)


def get_PSNR(I1, I2):
    s1 = cv2.absdiff(np.float32(I1), np.float32(I2)) #|I1 - I2|
    s1 = np.float32(s1)     # cannot make a square on 8 bits
    s1 = s1 * s1            # |I1 - I2|^2
    sse = s1.sum()          # sum elements per channel
    if sse <= 1e-10:        # sum channels
        return 0            # for small values return zero
    else:
        shape = I1.shape
        mse = 1.0 * sse / (shape[0] * shape[1] * shape[2])
        psnr = 10.0 * np.log10((255 * 255) / mse)
        return psnr

def get_MSSISM(I1, I2):
    C1 = 6.5025
    C2 = 58.5225
    # INITS
    I1 = np.float32(I1) # cannot calculate on one byte large values
    I2 = np.float32(I2)
    I2_2 = I2 * I2 # I2^2
    I1_2 = I1 * I1 # I1^2
    I1_I2 = I1 * I2 # I1 * I2
    # END INITS
    # PRELIMINARY COMPUTING
    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv2.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2
    sigma2_2 = cv2.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2
    sigma12 = cv2.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2
    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2                    # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2                    # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    ssim_map = cv2.divide(t3, t1)    # ssim_map =  t3./t1;
    mssim = cv2.mean(ssim_map)       # mssim = average of ssim map
    return mssim


def do_calculation(img1, img2):
    rmse = get_RMSE(img1, img2)
    psnr = get_PSNR(img1, img2)
    ssim = get_MSSISM(img1, img2)
    return rmse, psnr, ssim


def calculate_statistics_detail(elapsedTime, vid_est, vid_org, vid_bic, number_of_frames):
    stat_rmse = []
    stat_psnr = []
    stat_ssim = []
    for i in range(number_of_frames):
        current_stat_rmse, current_stat_psnr, current_stat_ssim = do_calculation(vid_est[i], vid_org[i])
        stat_rmse.append(current_stat_rmse)
        stat_psnr.append(current_stat_psnr)
        stat_ssim.append(current_stat_ssim)

    stat_interpolation_rmse = []
    stat_interpolation_psnr = []
    stat_interpolation_ssim = []
    for i in range(number_of_frames):
        current_stat_interpolation_rmse, current_stat_interpolation_psnr, current_stat_interpolation_ssim = \
            do_calculation(vid_bic[i], vid_org[i])
        stat_interpolation_rmse.append(current_stat_interpolation_rmse)
        stat_interpolation_psnr.append(current_stat_interpolation_psnr)
        stat_interpolation_ssim.append(current_stat_interpolation_ssim)

    stat = [elapsedTime, stat_rmse, stat_psnr, stat_ssim, stat_interpolation_rmse, stat_interpolation_psnr,
            stat_interpolation_ssim]
    return stat




