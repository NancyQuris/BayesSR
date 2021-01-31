import cv2


def calculation(I, I_sr): # to be filled
    prev_frame = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    next_frame = cv2.cvtColor(I_sr, cv2.COLOR_BGR2GRAY)
    u, v = cv2.calcOpticalFlowFarneback(prev_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return u, v
