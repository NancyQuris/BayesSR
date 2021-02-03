import cv2
import numpy as np


def calculation(I, I_sr):
    flow = cv2.calcOpticalFlowFarneback(np.float32(I), np.float32(I_sr), None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow[:, :, 0].reshape((I.shape[0], I.shape[1])), flow[:, :, 1].reshape((I.shape[0], I.shape[1]))
