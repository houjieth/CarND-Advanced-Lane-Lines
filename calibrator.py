import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


class Calibrator(object):
    def __init__(self, chessboard_w, chessboard_h):
        self.objp = np.zeros((chessboard_h * chessboard_w, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_w, 0:chessboard_h].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d points in real world space
        self.imgpoints = []  # 2d points in image plane.

        self.mtx = None
        self.dist = None

    def calibrate_with_chessboard_images(self, image_path):
        images = glob.glob(image_path)

        # Make a list of calibration images

        # Step through the list and search for chessboard corners
        for index, filename in enumerate(images):
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

            # If found, add object points, image points
            if ret is True:
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)

                # Draw and display the corners
                # cv2.drawChessboardCorners(img, (9,6), corners, ret)
                ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(
                    self.objpoints, self.imgpoints, (img.shape[1], img.shape[0]), None, None)

    def undistort(self, image):
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
