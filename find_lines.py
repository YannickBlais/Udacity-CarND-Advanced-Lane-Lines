import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# files and folders:
calibration_dir = "./camera_cal/"

# prepare object points
nx = 9
ny = 6


# Define a class to receive the characteristics of each line detection
class Line():
  def __init__(self):
    self.n_fits = 10
    self.clear_results()

  def clear_results(self):
    # was the line detected in the last iteration?
    self.detected = False
    # x values of the last n fits of the line
    self.recent_xfitted = []
    # average x values of the fitted line over the last n iterations
    self.bestx = None
    # polynomial coefficients averaged over the last n iterations
    self.best_fit = []
    # polynomial coefficients for the most recent fit
    self.current_fit = [np.array([False])]
    # radius of curvature of the line in some units
    self.radius_of_curvature = []
    # distance in meters of vehicle center from the line
    self.line_base_pos = []
    # difference in fit coefficients between last and new fits
    self.diffs = np.array([0,0,0], dtype='float')
    # x values for detected line pixels
    self.allx = None
    # y values for detected line pixels
    self.ally = None

  def add_result(self, radius_of_curvature, line_base_pos, left_fit, right_fit):
    self.detected = True
    self.radius_of_curvature.append(radius_of_curvature)
    self.line_base_pos.append(line_base_pos)
    self.best_fit.append([left_fit, right_fit])

    if len(self.radius_of_curvature) > self.n_fits:
      del self.radius_of_curvature[0]
      del self.line_base_pos[0]
      del self.best_fit[0]

  def get_avg_radius_of_curvature(self):
    return reduce(lambda x, y: x + y, self.radius_of_curvature) / len(self.radius_of_curvature)

  def get_avg_line_base_pos(self):
    return reduce(lambda x, y: x + y, self.line_base_pos) / len(self.line_base_pos)

  def get_best_fit(self):
    best_left_fit = [0, 0, 0]
    best_right_fit = [0, 0, 0]
    for i in range(0, len(self.best_fit)):
      best_left_fit[0] += self.best_fit[i][0][0]
      best_left_fit[1] += self.best_fit[i][0][1]
      best_left_fit[2] += self.best_fit[i][0][2]

      best_right_fit[0] += self.best_fit[i][1][0]
      best_right_fit[1] += self.best_fit[i][1][1]
      best_right_fit[2] += self.best_fit[i][1][2]

    best_left_fit[0] /= len(self.best_fit)
    best_left_fit[1] /= len(self.best_fit)
    best_left_fit[2] /= len(self.best_fit)

    best_right_fit[0] /= len(self.best_fit)
    best_right_fit[1] /= len(self.best_fit)
    best_right_fit[2] /= len(self.best_fit)

    return [best_left_fit, best_right_fit]


def display_images(src, dst, msg, map=None):
  f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
  f.tight_layout()
  ax1.imshow(src)
  ax1.set_title('Original Image', fontsize=50)
  if map is None:
    ax2.imshow(dst)
  else:
    ax2.imshow(dst, cmap=map)
  ax2.set_title(msg, fontsize=50)
  plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

def warper(img, src, dst):
  # Compute and apply perpective transform
  img_size = (img.shape[1], img.shape[0])
  M = cv2.getPerspectiveTransform(src, dst)
  warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

  return warped


def hls_select(img, frame_id, s_thresh=(0, 255), l_thresh=(0, 255), h_thresh=15):
  # 1) Convert to HLS color space
  hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
  H = hls[:, :, 0]
  L = hls[:, :, 1]
  S = hls[:, :, 2]
  # 2) Apply a threshold to the S channel
  binary = np.zeros_like(S)
  binary[(S > s_thresh[0]) & (S <= s_thresh[1])] = 1
  binary[(L > l_thresh[0]) & (L <= l_thresh[1])] = 1

  binary[(H < h_thresh)] = 0
  # 3) Return a binary image of threshold result

  save_image(np.array(L), 'L_channel', frame_id)
  save_image(np.array(S), 'S_channel', frame_id)
  save_image(np.array(H), 'H_channel', frame_id)
  return binary


def find_lanes(binary_warped, frame_id):
  # Assuming you have created a warped binary image called "binary_warped"
  # Take a histogram of the bottom half of the image
  histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
  # Create an output image to draw on and  visualize the result
  out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
  # Find the peak of the left and right halves of the histogram
  # These will be the starting point for the left and right lines
  midpoint = np.int(histogram.shape[0] / 2)
  leftx_base = np.argmax(histogram[:midpoint])
  rightx_base = np.argmax(histogram[midpoint:]) + midpoint

  # Choose the number of sliding windows
  nwindows = 9
  # Set height of windows
  window_height = np.int(binary_warped.shape[0] / nwindows)
  # Identify the x and y positions of all nonzero pixels in the image
  nonzero = binary_warped.nonzero()
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
  left_lane_inds = []
  right_lane_inds = []

  # Step through the windows one by one
  for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = binary_warped.shape[0] - (window + 1) * window_height
    win_y_high = binary_warped.shape[0] - window * window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
    cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
    nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
    nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
      leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:
      rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

  if len(right_lane_inds) == 0 or len(left_lane_inds) == 0:
    raise

  # Concatenate the arrays of indices
  left_lane_inds = np.concatenate(left_lane_inds)
  right_lane_inds = np.concatenate(right_lane_inds)

  # Extract left and right line pixel positions
  leftx = nonzerox[left_lane_inds]
  lefty = nonzeroy[left_lane_inds]
  rightx = nonzerox[right_lane_inds]
  righty = nonzeroy[right_lane_inds]

  # Fit a second order polynomial to each
  left_fit = np.polyfit(lefty, leftx, 2)
  right_fit = np.polyfit(righty, rightx, 2)

  # Generate x and y values for plotting
  ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
  left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
  right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

  out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
  out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

  save_image(out_img, 'lanes', frame_id)

  return ploty, left_fitx, right_fitx, left_fit, right_fit


def find_lines_from_known_lines(binary_warped, left_fit, right_fit, frame_id):
  # Assume you now have a new warped binary image
  # from the next frame of video (also called "binary_warped")
  # It's now much easier to find line pixels!
  nonzero = binary_warped.nonzero()
  nonzeroy = np.array(nonzero[0])
  nonzerox = np.array(nonzero[1])
  margin = 100
  left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
  nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
  right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
  nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

  # Again, extract left and right line pixel positions
  leftx = nonzerox[left_lane_inds]
  lefty = nonzeroy[left_lane_inds]
  rightx = nonzerox[right_lane_inds]
  righty = nonzeroy[right_lane_inds]
  # Fit a second order polynomial to each
  if len(lefty) == 0 or len(leftx) == 0 or len(rightx) == 0 or len(righty) == 0:
    return None, None, None, None, None
  left_fit = np.polyfit(lefty, leftx, 2)
  right_fit = np.polyfit(righty, rightx, 2)
  # Generate x and y values for plotting
  ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
  left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
  right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]


  out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
  out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
  out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

  save_image(out_img, 'lanes', frame_id)

  return ploty, left_fitx, right_fitx, left_fit, right_fit


def measure_curvature(leftx, rightx):
  # Generate some fake data to represent lane-line pixels
  ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
  quadratic_coeff = 3e-4  # arbitrary quadratic coefficient

  leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
  rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

  # Fit a second order polynomial to pixel positions in each fake lane line
  left_fit = np.polyfit(ploty, leftx, 2)
  left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
  right_fit = np.polyfit(ploty, rightx, 2)
  right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

  y_eval = np.max(ploty)
  left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
  right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

  left_min_y = left_fit[2] + y_eval * left_fit[1] + y_eval * left_fit[0] ** 2
  right_min_y = right_fit[2] + y_eval * right_fit[1] + y_eval * right_fit[0] ** 2

  ym_per_pix = 30.0 / 720  # meters per pixel in y dimension
  xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

  # Fit new polynomials to x,y in world space
  left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
  right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
  # Calculate the new radii of curvature
  left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
    2 * left_fit_cr[0])
  right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
    2 * right_fit_cr[0])
  # Now our radius of curvature is in meters
  left_min_y_m = left_min_y * xm_per_pix
  right_min_y_m = right_min_y * xm_per_pix
  pos_to_center = ((left_min_y_m + right_min_y_m) / 2.0) - (1280 * xm_per_pix / 2.0)

  return left_curverad, right_curverad, pos_to_center


def unwarp_and_project_lines(image, warped, Minv, ploty, left_fitx, right_fitx):
  # Create an image to draw the lines on
  warp_zero = np.zeros_like(warped).astype(np.uint8)
  color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

  # Recast the x and y points into usable format for cv2.fillPoly()
  pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
  pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
  pts = np.hstack((pts_left, pts_right))

  # Draw the lane onto the warped blank image
  cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

  # Warp the blank back to original image space using inverse perspective matrix (Minv)
  newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
  # Combine the result with the original image
  result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

  # display_images(image, result, 'Final result')

  return result


def add_text(image, avg_curvature, pos_to_center, frame_id):
  font = cv2.FONT_HERSHEY_SIMPLEX
  if avg_curvature < 2500:
    str = 'Radius of Curvature = {:>6}(m)'.format(int(avg_curvature))
  else:
    str = 'Radius of Curvature = [Inf]'
  cv2.putText(image, str, (10, 50), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

  if pos_to_center < 0.0:
    str = 'Vehicle is {:>+2.3f}(m) left of center'.format(pos_to_center)
  else:
    str = 'Vehicle is {:>+2.3f}(m) right of center'.format(pos_to_center)
  cv2.putText(image, str, (10, 100), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

  cv2.putText(image, 'Frame id: {}'.format(frame_id), (10, 150), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
  return image

save_images = False
def save_image(img, filename, frame_id):
  if not save_images:
    return
  debug_folder = './debug/'
  cv2.imwrite(debug_folder + filename + '_' + str(frame_id) + '.png', img)


def abs_sobel_thresh(img, frame_id, orient='x', thresh_min=0, thresh_max=255):

  # Apply the following steps to img
  # 1) Convert to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  # 2) Take the derivative in x or y given orient = 'x' or 'y'
  if orient == 'x':
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
  else:
    sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
  # 3) Take the absolute value of the derivative or gradient
  abs_sobel = np.absolute(sobel)
  # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
  scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
  # 5) Create a mask of 1's where the scaled gradient magnitude
  # is > thresh_min and < thresh_max
  binary_output = np.zeros_like(scaled_sobel)
  binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

  save_image(binary_output.copy()*255, 'sobel', frame_id)
  # 6) Return this mask as your binary_output image
  return binary_output


def perform_camera_calibration():
  single_objpoints = np.zeros((nx * ny, 3), np.float32)
  single_objpoints[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
  objpoints = []
  imgpoints = []
  for filename in os.listdir(calibration_dir):
    # Make a list of calibration images
    img = cv2.imread(calibration_dir + filename)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_shape = gray.shape[::-1]

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        objpoints.append(single_objpoints)

  calib_image = mpimg.imread('./camera_cal/calibration1.jpg')
  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(np.array(objpoints), np.array(imgpoints), img_shape, None, None)
  undistorted = cv2.undistort(calib_image, mtx, dist, None, mtx)
  cv2.imwrite("./output_images/undistort_output.png", undistorted)
  return mtx, dist


def main():
  mtx, dist = perform_camera_calibration()

  test_image = cv2.imread('./test_images/straight_lines1.jpg')
  hls_binary = hls_select(test_image, 0, s_thresh=(0, 1))

  src = np.float32([[250, 688], [585, 459], [701, 459], [1053, 688]])
  dst = np.float32([[200, 720], [200, 0], [1000, 0], [1000, 720]])
  warped = warper(test_image, src, dst)


  input_folder = './test_images/'
  video_folder = './videos/'

  # Define the codec and create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter('output_project_video.mp4', fourcc, 25.0, (1280, 720))

  line_state = Line()

  cap = cv2.VideoCapture(video_folder + 'project_video.mp4')
  i = 0

  start_debug = 0
  end_debug = 1000000
  while(cap.isOpened()):
      ret, straight_lines_image = cap.read()
      if straight_lines_image is None:
        break
      i += 1
      if i < start_debug or i > end_debug:
        continue

      undistorted = cv2.undistort(straight_lines_image, mtx, dist, None, mtx)
      save_image(undistorted, 'undistorted', i)


      hls_binary = hls_select(undistorted, i, s_thresh=(90, 250), l_thresh=(200, 255))
      save_image(hls_binary.copy()*255, 'hls_binary', i)

      sobel = abs_sobel_thresh(undistorted, i, orient='x', thresh_min=40, thresh_max=100)
      hls_binary = cv2.bitwise_or(hls_binary, sobel)
      # display_images(straight_lines_image, hls_binary, 'Warped and binarized', 'gray')

      src = np.float32([[250, 688], [585, 459], [701, 459], [1053, 688]])
      dst = np.float32([[200, 720], [200, 0], [1000, 0], [1000, 720]])
      top_down = warper(hls_binary, src, dst)

      save_image(top_down.copy()*255, 'top_down', i)

      if line_state.detected:
        best_fit = line_state.get_best_fit()
        ploty, left_fitx, right_fitx, left_fit, right_fit = \
          find_lines_from_known_lines(top_down, best_fit[0], best_fit[1], i)
        if ploty == None:
          line_state.detected = False
        else:
          left_curverad, right_curverad, pos_to_center = measure_curvature(left_fitx, right_fitx)
          avg_curverad = (left_curverad + right_curverad) / 2.0

          avg_line_base_pos = line_state.get_avg_line_base_pos()
          avg_radius_of_curvature = line_state.get_avg_radius_of_curvature()
          if abs(pos_to_center - avg_line_base_pos) < 0.5 and \
              ((avg_radius_of_curvature < 2500 and
               abs(avg_curverad - avg_radius_of_curvature) < 500.0) or
              avg_radius_of_curvature >= 2500):
            line_state.add_result(avg_curverad, pos_to_center, left_fit, right_fit)
          else:
            line_state.clear_results()
            print("Difference with previous fit too large, need to reinitialize fit."
                  " pos_to_center={}, line_base_pos={}, avg_curverad={}, radius_of_curvature={}"
                  .format(pos_to_center, avg_line_base_pos, avg_curverad, avg_radius_of_curvature))

      if not line_state.detected:
        try:
          ploty, left_fitx, right_fitx, left_fit, right_fit = find_lanes(top_down, i)
        except:
          line_state.clear_results()
          print("Error occured when finding lanes in image {}".format(i))
          continue

        left_curverad, right_curverad, pos_to_center = measure_curvature(left_fitx, right_fitx)
        avg_curverad = (left_curverad + right_curverad) / 2.0
        line_state.add_result(avg_curverad, pos_to_center, left_fit, right_fit)

      Minv = cv2.getPerspectiveTransform(dst, src)
      result = unwarp_and_project_lines(undistorted, top_down, Minv, ploty, left_fitx, right_fitx)

      add_text(result, line_state.get_avg_radius_of_curvature(), line_state.get_avg_line_base_pos(), i)

      out.write(result)
      save_image(result, 'result', i)


if __name__ == "__main__":
  main()