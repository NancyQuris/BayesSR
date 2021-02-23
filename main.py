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
blur_estimation = False
noise_estimation = True
show_image = False
save_result = True
make_video = False

video_path = './data/4 ( 01.36.24 )~( 01.37.24 ).avi'  # change to conduct SR for other videos
start_scene_index = 2
num_frames = 2

test_video = video_path
num_reference_frame = 1  # one side
scale_factor = 2  # one side
max_iteration = 5
eps = 0.0005  # stopping criteria
eps_out = 0.0001
eps_blur = 0.0001
eta = 0.02  # derivative of image
xi = 0.7  # derivative of kernel
alpha = 1  # noise
beta = 0.1  # noise
degrad = 1  # 0: image resize, 1: LPF + down sampling
kernel_size = 15  # degradation blur kernel
kernel_sigma = 1.2  # degradation blur kernel 1.2,1.6,2.0,2.4
noisesd = 0  # degradation (noise st. dev.) 0, 0.01, 0.03, 0.05

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
h_2d_sim = kernel_utility.create_gaussian_kernel(kernel_size, kernel_sigma)
h_1d, h_2d_init = kernel_utility.create_h(hsigma_init, kernel_size, hmode_init, high_res_width, high_res_height)
if blur_estimation:
    h_2d = h_2d_init
else:
    h_2d = h_2d_sim

b_original, g_original, r_original = frame_utility.split_bgr_frames(original_frames)

if degrad == 1:
    low_res_frames = frame_utility.create_low_res_with_S_K(original_frames, h_2d_sim, scale_factor, num_frames)
    bgr_low = frame_utility.add_noise(low_res_frames, 0, noisesd)
else:
    low_res_frames = low_res_frames_bic
    bgr_low = low_res_frames


bicubic_high_frames = frame_utility.resize(low_res_frames, scale_factor, cv2.INTER_CUBIC, True)
bgr_h_bicubic = frame_utility.resize(bgr_low, scale_factor, cv2.INTER_CUBIC, True)
B, G, R = frame_utility.split_bgr_frames(bgr_low)
B_init = frame_utility.create_initial_I(B, scale_factor)
G_init = frame_utility.create_initial_I(G, scale_factor)
R_init = frame_utility.create_initial_I(R, scale_factor)
B_sr, G_sr, R_sr = B_init, G_init, R_init

W0 = np.zeros((low_res_height, low_res_width))  # weight matrix for high resolution image
Ws = np.zeros((high_res_height, high_res_width))  # weight matrix for derivative
Wk = np.zeros((high_res_height, high_res_width))  # weight matrix for kernel
Wi = []  # weight matrix for high resolution image (neighbouring frames)
FI = []
u = []
v = []  # y-direction of optical flow
ut = []
vt = []
elapsed_time = []
for f_num in range(num_frames):
    Wi.append(np.zeros((low_res_height, low_res_width)))
    FI.append(np.zeros((high_res_height, high_res_width)))
    u.append(np.zeros((high_res_height, high_res_width)))
    v.append(np.zeros((high_res_height, high_res_width)))
    ut.append(np.zeros((high_res_height, high_res_width)))
    vt.append(np.zeros((high_res_height, high_res_width)))

for j in range(num_frames):
    print('SR No.%d frame' % j)
    n_back = min(maxnbf, j)
    n_for = min(maxnbf, num_frames - 1 - j)
    current_low = B
    current_init = B_init
    current_sr = B_sr
    for channel in range(3):
        if channel == 1:
            current_low = G
            current_init = G_init
            current_sr = G_sr
        elif channel == 2:
            current_low = R
            current_init = R_init
            current_sr = R_sr

        theta = np.ones((n_for + n_back + 1, 1))
        tic = time.perf_counter()  # start timer

        print("SR %d channel" % channel)
        # coordinate descent algorithm
        current_I = current_init[j]
        J0 = current_low[j]
        for i in range(-n_back, n_for + 1):
            FI[j + i] = current_init[j + i]

        for k in range(max_iteration):
            print('%d th iteration for frame %d...' % (k, j))
            I_old_out = current_I

            # estimate motion, IRLS, with optical flow algorithm
            if motion_estimation and optical_flow:
                print('motion estimation ...')
                for i in range(-n_back, n_for + 1):
                    if i == 0:
                        u[j + i] = np.zeros(current_I.shape)
                        v[j + i] = np.zeros(current_I.shape)
                        ut[j + i] = np.zeros(current_I.shape)
                        vt[j + i] = np.zeros(current_I.shape)
                    else:
                        u[j + i], v[j + i] = optical_flow_calculation.calculation(current_I, current_sr[j + i])
                        ut[j + i] = -u[j + i]
                        vt[j + i] = -v[j + i]
                        FI[j + i] = frame_utility.warped_img(current_I, u[j + i], v[j + i])

            # estimate noise
            nq = low_res_width * low_res_height
            print('noise estimation ...')
            if k == 0:
                for i in range(-n_back, n_for + 1):
                    theta[j + i] = max(1, max(n_back, n_for)) / (abs(i) + 1)
            else:
                for i in range(-n_back, n_for + 1):
                    if i == 0:
                        KI = frame_utility.cconv2d(h_2d, current_I)
                        SKI = frame_utility.down_sample(KI, scale_factor)
                        x_tmp = np.sum(np.abs(current_low[j + i] - SKI)) / nq
                        theta[j + i] = (alpha + nq - 1) / (beta + nq * x_tmp)
                    else:
                        KFI = frame_utility.cconv2d(h_2d, FI[j + i])
                        SKFI = frame_utility.down_sample(KFI, scale_factor)
                        x_tmp = np.sum(np.abs(current_low[j + i] - SKFI)) / nq
                        theta[j + i] = (alpha + nq - 1) / (beta + nq * x_tmp)

            # estimate high resolution image
            print('high resolution image estimation .. ')
            for m in range(max_iteration):
                print('%d th high resolution estimation IRLS iteration' % m)
                I_old_in = current_I
                W0 = kernel_utility.compute_W0(current_I, J0, h_2d, scale_factor)
                Ws = kernel_utility.compute_Ws(current_I)
                if motion_estimation:
                    for i in range(-n_back, n_for + 1):
                        #if i == 0:
                        FI[j + i] = frame_utility.warped_img(current_I, u[j + i], v[j + i])
                        Wi[j + i] = kernel_utility.compute_Wi(FI[j + i], current_low[j + i], h_2d, scale_factor)

                # estimate I
                if motion_estimation:
                    AI = kernel_utility.compute_Ax_h(current_I, W0, Ws, theta, h_2d, high_res_height, high_res_width,
                                                     scale_factor, eta, j, n_back, n_for, FI, Wi, ut, vt)
                    b = kernel_utility.compute_b_h(current_low, W0, theta, h_2d, scale_factor, high_res_height,
                                                   high_res_width, j, n_back, n_for, Wi, ut, vt)
                    current_I = kernel_utility.conj_gradient_himg(AI, current_I, b, W0, Ws, theta, h_2d, high_res_height,
                                                                  high_res_width,
                                                                  scale_factor, eta, j, n_back, n_for, FI, Wi, ut, vt,
                                                                  show_image)
                else:
                    AI = kernel_utility.compute_Ax(current_I, W0, Ws, theta[j], h_2d, scale_factor, eta)
                    b = kernel_utility.compute_b1(J0, W0, theta[j], h_2d, scale_factor)
                    current_I = kernel_utility.conj_gradient_himg_0(AI, current_I, b, W0, Ws, theta[j], h_2d, high_res_height,
                                                                    high_res_width, show_image, scale_factor, eta)
                diff_in = np.linalg.norm(current_I - I_old_in) / np.linalg.norm(I_old_in)
                if diff_in < eps:
                    if show_image:
                        print("break, diff_in < %f \n" % eps)
                        cv2.imshow('image', np.float32(current_I))
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                    break

            # estimate blur kernel, IRLS
            if blur_estimation:
                print('blur estimation...')
                K = frame_utility.otf2psf(frame_utility.psf2otf(h_2d, current_I.shape))
                for m in range(max_iteration): # originally max_iteration-2
                    print('%d th blur kernel estimation IRLS iteration' % m)
                    K_old = K
                    Wk = kernel_utility.compute_Wk(K, current_I, J0, scale_factor)
                    Ws = kernel_utility.compute_Ws(K)
                    # estimate kx
                    AK = kernel_utility.compute_Ax_k(K, Wk, Ws, theta[j], current_I, xi, scale_factor)
                    b = kernel_utility.compute_b1_k(J0, Wk, theta[j], current_I, scale_factor)
                    K = kernel_utility.conj_gradient_kernel(AK, K, b, Wk, Ws, theta[j], current_I, high_res_height,
                                                            high_res_width,
                                                            kernel_size, xi, scale_factor)
                    if np.linalg.norm(K_old) != 0:
                        diff_K = np.linalg.norm(K - K_old) / np.linalg.norm(K_old)
                    else:
                        diff_K = np.linalg.norm(K - K_old)
                    if diff_K < eps_blur:
                        if show_image:
                            print('break, diff_K < %f\n' % eps_blur)
                        break
                half = kernel_size // 2
                h_2d = K[int(high_res_height / 2 - half + 1):int(high_res_height / 2 + half + 1),
                       int(low_res_width / 2 - half + 1):int(low_res_width / 2 + half + 1)]
                if np.sum(h_2d) != 0:
                    h_2d = h_2d / np.sum(h_2d)

            # check convergence
            diff_out = np.linalg.norm(current_I - I_old_out) / np.linalg.norm(I_old_out)
            if diff_out < eps_out:
                print('converge! norm(I-I_old)/norm(I_old) < %f\n' % eps_out)
                break

        elapsed_time.append(time.perf_counter() - tic)
        current_sr[j] = current_I

# statistic results, Bayesian method
video_result = frame_utility.merge(B_sr, G_sr, R_sr)
# if degrad == 0:
#     video_result = frame_utility.shift_adjust(video_result, scale_factor, -1)
# if degrad == 1:
#     bicubic_high_frames = frame_utility.shift_adjust(bicubic_high_frames, scale_factor, 1)

# estimated blur kernel
if blur_estimation:
    h_2d_est = h_2d  # original is h_2d_est[:, :, 1] = h_2d
    if show_image:
        cv2.imshow('est blur kernel', h_2d)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# statistics
stat = frame_utility.calculate_statistics_detail(elapsed_time, video_result, original_frames, bicubic_high_frames, num_frames)

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

