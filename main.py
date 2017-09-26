import os
import cv2

from calibrator import Calibrator
from image_processor import ImageProcessor
from lane_finder import LaneFinder

if __name__ == '__main__':
    # Calibrate camera
    calibrator = Calibrator(9, 6)
    calibrator.calibrate_with_chessboard_images('camera_cal/*.jpg')

    # Test calibration result
    # img = cv2.imread('camera_cal/calibration1.jpg')
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.imwrite('images/original.jpg', img)
    # img = calibrator.undistort(img)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.imwrite('images/undistorted.jpg', img)

    image_processor = ImageProcessor()

    #lane_finder = LaneFinder(calibrator, image_processor)

    #lane_finder.find_lane_from_image('test_images/test5.jpg', 'output_images')

    for filename in os.listdir('test_images'):
        if filename.endswith('.jpg'):
            lane_finder.find_lane_from_image(os.path.join('test_images', filename), 'output_images')

    lane_finder.find_lane_from_video('test_videos/project_video.mp4', 'output_videos')
