import numpy as np
import cv2


class ImageProcessor(object):

    def __init__(self):
        src = np.float32(
            [[260, 691],
             [602, 446],
             [683, 446],
             [1049, 682]])

        dst = np.float32(
            [[260, 691],
             [260, 0],
             [1049, 0],
             [1049, 682]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.M_inv = cv2.getPerspectiveTransform(dst, src)

    def warp(self, image):
        return cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    def unwarp(self, image):
        return cv2.warpPerspective(image, self.M_inv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    def to_binary_edge(self, img):
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        s_channel = hls[:, :, 2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Filter by lightness (this is only for filter out black areas like shadow)
        l_channel = hls[:, :, 1]
        l_threshold_min = 50
        l_binary = np.zeros_like(s_channel)

        l_binary[(l_channel >= l_threshold_min)] = 1

        # cv2.imshow('img', l_binary * 255)
        # cv2.waitKey(0)

        # Filter by saturation
        s_threshold_min = 170
        s_threshold_max = 255
        s_binary = np.zeros_like(s_channel)

        s_binary[(s_channel >= s_threshold_min) & (s_channel <= s_threshold_max)] = 1

        # cv2.imshow('img', s_binary * 255)
        # cv2.waitKey(0)

        # Filter by gradient at X axis

        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        sobel_threshold_min = 20
        sobel_threshold_max = 100

        sx_binary = np.zeros_like(gray)
        sx_binary[(scaled_sobelx >= sobel_threshold_min) & (scaled_sobelx <= sobel_threshold_max)] = 1

        # cv2.imshow('img', sx_binary * 255)
        # cv2.waitKey(0)

        # Combine both filter results
        combined_binary = np.zeros_like(gray)
        combined_binary[((s_binary == 1) | (sx_binary == 1)) & (l_binary == 1)] = 1
        return combined_binary
