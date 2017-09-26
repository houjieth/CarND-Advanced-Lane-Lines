import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

from collections import deque

SMOOTH_FRAMES = 7


class LaneFinder(object):
    def __init__(self, calibrator, image_processor):
        self.calibrator = calibrator
        self.image_processor = image_processor
        self.left_fits = deque()  # left_fit history
        self.right_fits = deque()  # right_fit history
        self.left_fit = None
        self.right_fit = None

    def calc_lane(self, image):
        # Take a histogram of the bottom half the image
        histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((image, image, image)) * 255

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9

        # Set height of windows
        window_height = np.int(image.shape[0] / nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()

        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Set the width of the windows +/- margin
        margin = 100

        # Set minimum number of pixels found to recenter window
        minpix = 50

        # Create empty lists to receive left and right lane pixel indices
        left_lane_idxs = []
        right_lane_idxs = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in xclclip.write_videofile(white_output, audio=False)ip.write_videofile(white_output, audio=False) and y (and right and left)
            win_y_low = image.shape[0] - (window + 1) * window_height
            win_y_high = image.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_idxs = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
            good_right_idxs = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_idxs.append(good_left_idxs)
            right_lane_idxs.append(good_right_idxs)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_idxs) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_idxs]))
            if len(good_right_idxs) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_idxs]))

        # Concatenate the arrays of indices
        left_lane_idxs = np.concatenate(left_lane_idxs)
        right_lane_idxs = np.concatenate(right_lane_idxs)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_idxs]
        lefty = nonzeroy[left_lane_idxs]
        rightx = nonzerox[right_lane_idxs]
        righty = nonzeroy[right_lane_idxs]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Smooth the fitting polynomial parameters by looking at previous frame's results
        self.smooth_fit_parameters(left_fit, right_fit)

        # Generate x and y values for plotting
        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        out_img[nonzeroy[left_lane_idxs], nonzerox[left_lane_idxs]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_idxs], nonzerox[right_lane_idxs]] = [0, 0, 255]

        # cv2.imshow('img', out_img)
        # cv2.waitKey(0)
        # cv2.imwrite('images/pipeline_polyfit.jpg', out_img)

        # plt.imshow(out_img)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        # plt.show()

        # Estimate lane curvature and car's offset to lane center
        y_eval = np.max(ploty)
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 300  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 900  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

        # Calculate the new rad of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

        # Now our radius of curvature is in meters
        # print(left_curverad, 'm', right_curverad, 'm')

        # Estimate car's offset to lane center
        lane_left_bottom_y = image.shape[0] - 1
        lane_left_bottom_x = self.left_fit[0] * lane_left_bottom_y ** 2 + self.left_fit[1] * lane_left_bottom_y + self.left_fit[2]

        lane_right_bottom_y = image.shape[0] - 1
        lane_right_bottom_x = self.right_fit[0] * lane_right_bottom_y ** 2 + self.right_fit[1] * lane_right_bottom_y + self.right_fit[2]

        # print(lane_left_bottom_x)
        # print(lane_left_bottom_y)

        lane_middle_x = lane_left_bottom_x * 0.5 + lane_right_bottom_x * 0.5
        car_lane_middle_offset = abs(lane_middle_x - image.shape[1] / 2) * xm_per_pix

        # print(lane_middle_x)
        # print(image.shape[1] / 2)
        # print(car_lane_middle_offset)

        return left_fitx, right_fitx, ploty, left_curverad, right_curverad, car_lane_middle_offset

    def smooth_fit_parameters(self, left_fit, right_fit):
        if len(self.left_fits) == SMOOTH_FRAMES:
            self.left_fits.popleft()
        self.left_fits.append(left_fit)

        if len(self.right_fits) == SMOOTH_FRAMES:
            self.right_fits.popleft()
        self.right_fits.append(right_fit)

        if self.left_fit is None:
            self.left_fit = left_fit
            self.right_fit = right_fit
            return

        # Use O(5) instead of O(1) for better readability
        self.left_fit[0] = sum(fit[0] for fit in self.left_fits) / SMOOTH_FRAMES
        self.left_fit[1] = sum(fit[1] for fit in self.left_fits) / SMOOTH_FRAMES
        self.left_fit[2] = sum(fit[2] for fit in self.left_fits) / SMOOTH_FRAMES
        
        self.right_fit[0] = sum(fit[0] for fit in self.right_fits) / SMOOTH_FRAMES
        self.right_fit[1] = sum(fit[1] for fit in self.right_fits) / SMOOTH_FRAMES
        self.right_fit[2] = sum(fit[2] for fit in self.right_fits) / SMOOTH_FRAMES

    def draw_lane_on_image(self, undist, warped, left_fitx, right_fitx, ploty):
        # Create an image to draw the lines on
        color_warp = np.zeros_like(warped).astype(np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        unwarp_lane = self.image_processor.unwarp(color_warp)

        # Combine the result with the original image
        return cv2.addWeighted(undist, 1, unwarp_lane, 0.3, 0)

    def draw_info_on_image(self, image, left_curvature, right_curvature, car_lane_middle_offset):
        font = cv2.FONT_HERSHEY_SIMPLEX
        curvature_text = "Curvature Radius(m): " + str(left_curvature * 0.5 + right_curvature * 0.5)
        cv2.putText(image, curvature_text, (10, 30), font, 1, (255, 255, 255), 2)
        offset_text = "Car's offset to lane center(m): " + str(car_lane_middle_offset)
        cv2.putText(image, offset_text, (10, 60), font, 1, (255, 255, 255), 2)
        return image

    def find_lane(self, image):
        # cv2.imwrite('images/pipeline_original.jpg', image)

        # Undistort image
        undist = self.calibrator.undistort(image)
        # cv2.imwrite('images/pipeline_undistorted.jpg', undist)

        # Perspective transform image
        warped = self.image_processor.warp(undist)
        # cv2.imshow('img', warped)
        # cv2.waitKey(0)
        # cv2.imwrite('images/pipeline_warped.jpg', warped)

        # Edge detection into binary image
        binary = self.image_processor.to_binary_edge(warped)
        # cv2.imshow('img', binary * 255)
        # cv2.waitKey(0)
        # cv2.imwrite('images/pipeline_binary.jpg', binary * 255)

        # Fit polynomial line
        left_fitx, right_fitx, ploty, left_curvature, right_curvature, car_lane_middle_offset = self.calc_lane(binary)

        # Draw line back to image
        lane_image = self.draw_lane_on_image(undist, warped, left_fitx, right_fitx, ploty)
        cv2.imwrite('images/pipeline_lane.jpg', lane_image)

        # Draw curvature on image
        result = self.draw_info_on_image(lane_image, left_curvature, right_curvature, car_lane_middle_offset)

        return result

    def find_lane_from_image(self, image_path, output_path):
        image = cv2.imread(image_path)
        result = self.find_lane(image)

        filename = os.path.basename(image_path)
        cv2.imwrite(os.path.join(output_path, filename), result)

    def find_lane_from_video(self, video_path, output_path):
        video = VideoFileClip(video_path)
        result = video.fl_image(self.find_lane)

        filename = os.path.basename(video_path)
        result.write_videofile(os.path.join(output_path, filename), audio=False)

