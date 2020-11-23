import cv2
import os
import numpy as np
import frame_utility
import kernel_utility

####################################
#
# Implementation based on code from
# https://github.com/seunghwanyoo/bayesian_vid_sr/blob/master/main_BayesianSR_city.m
#
####################################


# global parameters
motion_estimation = 1
optical_flow = 1
blur_estimation = 1
noise_estimation = 1

video_path = './data'
start_scene_index = 0
num_frames = 5

test_video = video_path
num_reference_frame = 2  # one side
scale_factor = 4
max_iteration = 5
eps = 0.0005  # stopping criteria
eps_out = 0.0001
eps_blur = 0.0001
eta = 0.02  # derivative of image
xi = 0.7  # derivative of kernel
alpha = 1  # noise
beta = 0.1  # noise
degrad = 1  # 0: imresize, 1: LPF + downsampling
hsize = 15  # degradation blur kernel
hsigma = 1.6  # degradation blur kernel 1.2,1.6,2.0,2.4
noisesd = 0  # degradation (noise st. dev.) 0, 0.01, 0.03, 0.05
# initialise estimated blur kernel
hmode_init = 1  # 1: Gaussian, 2: uniform
hsigma_init = 2.0

# cut frames
frame_utility.frame_cutting(video_path)
# load frames
original_frames, low_res_frames_bic = frame_utility.load_frame(video_path, start_scene_index, num_frames, 1 / scale_factor)
high_res_width, high_res_height = original_frames[0].shape[1], original_frames[0].shape[0]
low_res_width, low_res_height = low_res_frames_bic[0].shape[1], low_res_frames_bic[0].shape[0]

# generate gaussian kernel
h_2d_sim = kernel_utility.fspecial_gaussian((hsize, hsize), hsigma)

if blur_estimation:
    h_1d, h_2d_init = kernel_utility.create_h(hsigma_init, hsize, hmode_init, high_res_width, high_res_height)
    h_2d = h_2d_init
else:
    h_2d = h_2d_sim

ycrcb_h_orignial = frame_utility.bgr2ycrcb(original_frames)
y_original, Cr_original, Cb_original = frame_utility.split_ycrcb_frames(ycrcb_h_orignial)

if degrad == 1:
    low_res_frames = frame_utility.create_low_res_with_S_K(original_frames, h_2d_sim, scale_factor, num_frames)
    ycrcb_low = frame_utility.bgr2ycrcb(low_res_frames)
    ycrcb_low = frame_utility.add_noise(ycrcb_low, 0, noisesd * noisesd)
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

for i in range(num_frames):
    print("SR %d frame" % i)
