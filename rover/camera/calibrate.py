import cv2
import numpy as np

import cv2 as cv
import glob
import numpy as np


def calibrate_camera(image):
    images = []

    im = cv.imread(image, 1)
    images.append(im)
    print(f"Images len {len(images)}")

    # criteria used by checkerboard pattern detector.
    # Change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    rows = 7  # number of checkerboard rows.
    columns = 7  # number of checkerboard columns.
    world_scaling = 1.  # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints = []  # 2d points in image plane.

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:
            # Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            # opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
            cv.imshow('img', frame)
            k = cv.waitKey(500)

            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', mtx)
    print('distortion coeffs:', dist)
    print('Rs:\n', rvecs)
    print('Ts:\n', tvecs)

    return mtx, dist


mtx1, dist1 = calibrate_camera('calibrate1.png')
mtx2, dist2 = calibrate_camera('calibrate3.png')


















# set the number of calibration images
n_images = 10

# set the size of the calibration pattern
pattern_size = (7, 7)

# prepare the object points for the calibration pattern
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

# create arrays to store the object points and image points for each camera
objpoints = []
imgpoints_left = []
imgpoints_right = []

# loop over the calibration images
# for i in range(n_images):
    # load the left and right images
img_left = cv2.imread(f'J2/calibrate1.png')
img_right = cv2.imread(f'D2/calibrate3.png')
# convert the images to grayscale
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

# detect the corners of the calibration pattern in both images
ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size)
ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size)
print(f"Ret left {ret_left} corner left {corners_left.shape}")
print(f"Ret right {ret_right} corner right {corners_right.shape}")
print(corners_left)
# if both corners are detected, add the object points and image points to the lists
if ret_left and ret_right:
    objpoints.append(objp)
    imgpoints_left.append(corners_left)
    imgpoints_right.append(corners_right)

# perform stereo calibration
ret, K_left, D_left, K_right, D_right, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1, mtx2, dist2, gray_left.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC)
print(K_left)
print(D_left)
print(K_right)
print(D_right)
print(R)
print(T)
print(E)
print(F)
# compute the rectification transforms
# # R_left, R_right, P_left, P_right, Q, _, _ = cv2.stereoRectify(K_left, D_left, K_right, D_right, gray_left.shape[::-1], R, T)
#
# # save the calibration parameters to file
# np.savez('calibration.npz', K_left=K_left, D_left=D_left, K_right=K_right, D_right=D_right, R_left=R_left, R_right=R_right, P_left=P_left, P_right=P_right, Q=Q)
pt = np.array([[2], [100], [1]])  # (x, y) coordinates of the point

for i in range(0,540,20):
    pt[0][0] = i
    pt[1][0] = i+20
    if ret_left:
        # Draw the point on the image
        p = (int(pt[0][0]),int(pt[1][0]))
        img_left = cv2.circle(img_left, p, 3, (0, 0, 255*i/640), -1)
        fnl = cv2.drawChessboardCorners(img_left, (7, 7), corners_left, ret_left)

    if ret_right:
        pt2 = R.dot(pt) + T
        print(f" From {pt} tp {pt2}")
        p = (int(pt2[0][0]),int(pt2[1][0]))
        img_right = cv2.circle(img_right, p, 3, (0, 0, 255*i/640), -1)
        fnl2 = cv2.drawChessboardCorners(img_right, (7, 7), corners_right, ret_right)

while True:
    cv2.imshow("fnl", fnl)
    cv2.imshow("fnl2", fnl2)
    key = cv2.waitKey(0)
    if key == ord('q'):
        # Quit the program if 'q' key is pressed
        pass

