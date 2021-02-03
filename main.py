import cv2
import datetime
import numpy as np
import os
import time
import frame_utility
import kernel_utility
import optical_flow_calculation

####################################
#
# Implementation based on code from
# https://github.com/seunghwanyoo/bayesian_vid_sr/blob/master/main_BayesianSR_city.m
#
####################################


# global parameters
motion_estimation = True
optical_flow = True
blur_estimation = True
noise_estimation = True
show_image = False
save_result = True
make_video = False

video_path = './data/4 ( 01.36.24 )~( 01.37.24 ).avi'  # change to conduct SR for other videos
start_scene_index = 0
num_frames = 3

test_video = video_path
num_reference_frame = 1  # one side
scale_factor = 2  # one side
max_iteration = 1
eps = 0.0005  # stopping criteria
eps_out = 0.0001
eps_blur = 0.0001
eta = 0.02  # derivative of image
xi = 0.7  # derivative of kernel
alpha = 1  # noise
beta = 0.1  # noise
degrad = 1  # 0: image resize, 1: LPF + down sampling
hsize = 15  # degradation blur kernel
hsigma = 1.6  # degradation blur kernel 1.2,1.6,2.0,2.4
noisesd = 0.01  # degradation (noise st. dev.) 0, 0.01, 0.03, 0.05

# initialise estimated blur kernel
hmode_init = 1  # 1: Gaussian, 2: uniform
hsigma_init = 2.0
maxnbf = min(num_frames, num_reference_frame)  # maximum frame numbers for reference

# cut frames
# frame_utility.frame_cutting(video_path)
# load frames
original_frames, low_res_frames_bic = frame_utility.load_frame(video_path, start_scene_index, num_frames, 1/scale_factor)
high_res_width, high_res_height = original_frames[0].shape[1], original_frames[0].shape[0]
low_res_width, low_res_height = low_res_frames_bic[0].shape[1], low_res_frames_bic[0].shape[0]

# generate gaussian kernel
h_2d_sim = kernel_utility.create_gaussian_kernel(hsize, hsigma)

if blur_estimation:
    h_1d, h_2d_init = kernel_utility.create_h(hsigma_init, hsize, hmode_init, high_res_width, high_res_height)
    h_2d = h_2d_init
else:
    h_2d = h_2d_sim

ycrcb_h_orignial = frame_utility.bgr2ycrcb(original_frames)
y_original, Cr_original, Cb_original = frame_utility.split_ycrcb_frames(ycrcb_h_orignial)

if degrad == 1:
    low_res_frames = frame_utility.create_low_res_with_S_K(original_frames, h_2d_sim, 1/scale_factor, num_frames)
    ycrcb_low = frame_utility.bgr2ycrcb(low_res_frames)
    ycrcb_low = frame_utility.add_noise(ycrcb_low, 0, noisesd)
else:
    low_res_frames = low_res_frames_bic
    ycrcb_low = frame_utility.bgr2ycrcb(low_res_frames_bic)

bicubic_high_frames = frame_utility.resize(low_res_frames, scale_factor, cv2.INTER_CUBIC, True)
ycrcb_h_bicubic = frame_utility.resize(ycrcb_low, scale_factor, cv2.INTER_CUBIC, True)

J, Cr_low, Cb_low = frame_utility.split_ycrcb_frames(ycrcb_low)
I_init = frame_utility.create_initial_I(J, scale_factor)

W0 = np.zeros((low_res_height, low_res_width))  # weight matrix for high resolution image
Ws = np.zeros((high_res_height, high_res_width))  # weight matrix for derivative
Wk = np.zeros((high_res_height, high_res_width))  # weight matrix for kernel
Wi = []  # weight matrix for high resolution image (neighbouring frames)
I_sr = I_init  # initialization of super resolved image

FI = []  # what FI is used for? and how to create a array in a right way
u = []
v = []  # y-direction of optical flow
ut = []
vt = []
theta = []
for j in range(num_frames):
    Wi.append(np.zeros((low_res_height, low_res_width)))
    FI.append(np.zeros((high_res_height, high_res_width)))
    u.append(np.zeros((high_res_height, high_res_width)))
    v.append(np.zeros((high_res_height, high_res_width)))
    ut.append(np.zeros((high_res_height, high_res_width)))
    vt.append(np.zeros((high_res_height, high_res_width)))
    theta.append(0)

tic = time.perf_counter()  # start timer
elapsed_time = 0

for j in range(num_frames):
    print('SR No.%d frame' % j)
    n_back = min(maxnbf, j)
    n_for = min(maxnbf, num_frames - 1 - j)

    # coordinate descent algorithm
    I = I_init[j]
    J0 = J[j]
    for i in range(-n_back, n_for + 1):
        FI[j + i] = I_init[j + i]

    for k in range(max_iteration):
        print('%d th iteration for frame %d...' % (k, j))
        I_old_out = I

        # estimate motion, IRLS, with optical flow algorithm
        if motion_estimation and optical_flow:
            print('motion estimation ...')
            for i in range(-n_back, n_for + 1):
                if i == 0:
                    u[j + i] = np.zeros(I.shape)
                    v[j + i] = np.zeros(I.shape)
                    ut[j + i] = np.zeros(I.shape)
                    vt[j + i] = np.zeros(I.shape)
                else:
                    u[j + i], v[j + i] = optical_flow_calculation.calculation(I, I_sr[j + i])
                    ut[j + i] = -u[j + i]
                    vt[j + i] = -v[j + i]
                    FI[j + i] = frame_utility.warped_img(I, u[j + i], v[j + i])

        # estimate noise
        nq = low_res_width * low_res_height
        print('noise estimation ...')
        if k == 1:
            for i in range(-n_back, n_for + 1):
                theta[j + i] = max(1, max(n_back, n_for)) / (abs(i) + 1)
        else:
            for i in range(-n_back, n_for + 1):
                if i == 0:
                    KI = frame_utility.cconv2d(h_2d, I)
                    SKI = frame_utility.down_sample(KI, scale_factor)
                    x_tmp = np.sum(np.abs(J[j + i] - SKI)) / nq
                    theta[j + i] = (alpha + nq - 1) / (beta + nq * x_tmp)
                else:
                    KFI = frame_utility.cconv2d(h_2d, FI[j + i])
                    SKFI = frame_utility.down_sample(KFI, scale_factor)
                    x_tmp = np.sum(np.abs(J[j + i] - SKFI)) / nq
                    theta[j + i] = (alpha + nq - 1) / (beta + nq * x_tmp)

        # estimate high resolution image
        print('high resolution image estimation .. ')
        for m in range(max_iteration):
            print('%d th high resolution estimation IRLS iteration' % m)
            I_old_in = I
            W0 = kernel_utility.compute_W0(I, J0, h_2d, scale_factor)
            Ws = kernel_utility.compute_Ws(I)
            if motion_estimation:
                for i in range(-n_back - 1, n_for):
                    if i == 0:
                        FI[j + i] = frame_utility.warped_img(I, u[j + i], v[j + i])
                        Wi[j + i] = kernel_utility.compute_Wi(FI[j + i], J[j + i], h_2d, scale_factor)

            # estimate I
            # AI = np.zeros((high_res_height, high_res_width))
            # b = np.zeros((high_res_height, high_res_width))
            if motion_estimation:
                AI = kernel_utility.compute_Ax_h(I, W0, Ws, theta, h_2d, high_res_height, high_res_width, scale_factor,
                                                 eta, j, n_back, n_for, FI, Wi, ut, vt)
                b = kernel_utility.compute_b_h(J, W0, theta, h_2d, scale_factor, high_res_height, high_res_width, j,
                                               n_back, n_for, Wi, ut, vt)
                I = kernel_utility.conj_gradient_himg(AI, I, b, W0, Ws, theta, h_2d, high_res_height, high_res_width,
                                                      scale_factor, eta, j, n_back, n_for, FI, Wi, ut, vt, show_image)
            else:
                AI = kernel_utility.compute_Ax(I, W0, Ws, theta, h_2d, scale_factor, eta)
                b = kernel_utility.compute_b1(J0, W0, theta[j], h_2d, scale_factor)
                I = kernel_utility.conj_gradient_himg_0(AI, I, b, W0, Ws, theta[j], h_2d, high_res_height,
                                                        high_res_width, show_image, scale_factor, eta)
            diff_in = np.linalg.norm(I - I_old_in) / np.linalg.norm(I_old_in)
            if diff_in < eps:
                if show_image:
                    print("break, diff_in < %f \n" % eps)
                    cv2.imshow('image', np.float32(I))
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                break

        # estimate blur kernel, IRLS
        if blur_estimation:
            print('blur estimation...')
            psf_k = frame_utility.psf2otf(h_2d, I.shape)
            K = frame_utility.otf2psf(psf_k, I.shape)
            for m in range(max_iteration - 2):
                print('%d th blur kernel estimation IRLS iteration' % m)
                K_old = K
                Wk = kernel_utility.compute_Wk(K, I, J0, scale_factor)
                Ws = kernel_utility.compute_Ws(K)
                # estimate kx
                AK = kernel_utility.compute_Ax_k(K, Wk, Ws, theta[j], I, xi, scale_factor)
                b = kernel_utility.compute_b1_k(J0, Wk, theta[j], I, scale_factor)
                k = kernel_utility.conj_gradient_kernel(AK, K, b, Wk, Ws, theta[j], I, high_res_height, high_res_width,
                                                        hsize, xi, scale_factor)
                diff_K = np.linalg.norm(K - K_old) / np.linalg.norm(K_old)
                if diff_K < eps_blur:
                    if show_image:
                        print('break, diff_K < %f\n' % eps_blur)
                    break
            half = hsize // 2
            h_2d = K[int(high_res_height / 2 - half + 1):int(high_res_height / 2 + half + 1),
                   int(low_res_width / 2 - half + 1):int(low_res_width / 2 + half + 1)]
            h_2d = h_2d / np.sum(h_2d)

        # check convergence
        diff_out = np.linalg.norm(I - I_old_out) / np.linalg.norm(I_old_out)
        if diff_out < eps_out:
            print('converge! norm(I-I_old)/norm(I_old) < %f\n' % eps_out)

    elapsed_time = time.perf_counter() - tic
    I_sr[j] = I

# statistic results, Bayesian method
video_result = frame_utility.y2rgb(I_sr, ycrcb_h_bicubic)
if degrad == 0:
    video_result = frame_utility.shift_adjust(video_result, scale_factor, -1)
# cv2.imshow('sr bayesian (frame1)', video_result[1])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# bicubic method
if degrad == 1:
    bicubic_high_frames = frame_utility.shift_adjust(bicubic_high_frames, scale_factor, 1)
# cv2.imshow('sr bicubic (frame1)', bicubic_high_frames[1])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# estimate blur kernel
if blur_estimation:
    h_2d_est = h_2d  # original is h_2d_est[:, :, 1] = h_2d
    if show_image:
        cv2.imshow('est blur kernel', h_2d)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# statistics
stat = frame_utility.calculate_statistics_detail(elapsed_time, video_result, original_frames, bicubic_high_frames,
                                                 num_frames)

# save results
file_name = 'bayesian'
result_path = 'Result_' + datetime.datetime.today().strftime('%m%d%Y')
os.mkdir(result_path)

count = 0
for v in video_result:
    cv2.imwrite(os.getcwd() + "/" + result_path + "result%d.jpg" % count, v)
    count += 1

count_2 = 0
for v2 in bicubic_high_frames:
    cv2.imwrite(os.getcwd() + "/" + result_path + "bicubic%d.jpg" % count_2, v2)
    count_2 += 1

statistics_name = ['elapsedTime', 'rmse', 'psnr', 'ssim', 'interpolation_rmse', 'interpolation_psnr',
            'interpolation_ssim']

with open("result.txt", 'w') as output:
    for i in range(len(stat)):
        output.write(statistics_name[i] + ":\n")
        output.write(str(stat[i]) + '\n')

if make_video:
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 15.0, video_result[0].shape)
    for v in video_result:
        out.write(v)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()

