import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt
from helpers.cv_utils import concat_images
import random
ROWS = 7
COLS = 7

def resize_image(img):
    height, width, channels = img.shape

    # Check if the image is already 640x480
    if width != 640 or height != 480:
        # Resize the image to 640x480 using linear interpolation
        img = cv.resize(img, (640, 480), interpolation=cv.INTER_LINEAR)
    return img
def calibrate_camera(images_folder):
    images_names = glob.glob(images_folder)
    images_names = sorted(images_names)
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        im = resize_image(im)

        images.append(im)

    # plt.figure(figsize = (10,10))
    # ax = [plt.subplot(2,2,i+1) for i in range(4)]
    #
    # for a, frame in zip(ax, images):
    #     a.imshow(frame[:,:,[2,1,0]])
    #     a.set_xticklabels([])
    #     a.set_yticklabels([])
    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.show()

    # criteria used by checkerboard pattern detector.
    # Change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = ROWS  # number of checkerboard rows.
    columns = COLS  # number of checkerboard columns.
    world_scaling =  2.6  # change this to the real world square size. Or not.

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
            while True:
                cv.imshow('img', frame)
                key = cv.waitKey(0)
                if key == ord('q'):
                    # Quit the program if 'q' key is pressed
                    break

            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', mtx)
    print('distortion coeffs:', dist)
    # print('Rs:\n', rvecs)
    # print('Ts:\n', tvecs)
    np.save('rvecs2.npy',rvecs)
    np.save('tvecs2.npy',tvecs)
    return mtx, dist


def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder):
    np.save('mtx2.npy',mtx2)
    np.save('dist2.npy',dist2)
    # read the synched frames
    images_names = glob.glob(frames_folder)
    images_names = sorted(images_names)
    c1_images_names = []
    c2_images_names = []
    for img in images_names:
        if 'phone' in img:
            c2_images_names.append(img)
        else:
            c1_images_names.append(img)
    c1_images_names = sorted(c1_images_names)
    c2_images_names = sorted(c2_images_names)
    print(f"C1 images names {c1_images_names} c2 images names {c2_images_names}")
    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        _im = resize_image(_im)
        c1_images.append(_im)

        _im = cv.imread(im2, 1)
        _im = resize_image(_im)
        c2_images.append(_im)

    # change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)


    rows = ROWS  # number of checkerboard rows.
    columns = COLS  # number of checkerboard columns.
    world_scaling = 2.6  # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (ROWS, COLS), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (ROWS, COLS), None)

        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            cv.drawChessboardCorners(frame1, (rows, columns), corners1, c_ret1)


            cv.drawChessboardCorners(frame2, (rows, columns), corners2, c_ret2)
            while True:
                undist_frame2 = cv.undistort(frame2,mtx2,dist2)
                final_image = concat_images([frame1, frame2, undist_frame2], ['Frame1', 'Frame2', 'undist_frame2'])
                cv.imshow('img', final_image)
                key = cv.waitKey(0)
                if key == ord('q'):
                    # Quit the program if 'q' key is pressed
                    break

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1,
                                                                 dist1,
                                                                 mtx2, dist2, (width, height), criteria=criteria,
                                                                 flags=stereocalibration_flags)

    print(f"Stereo calibrate RMS ret {ret}")
    return R, T


def triangulate(mtx1, mtx2, R, T):

    frame1 = cv.imread('data/gray_5.png')
    frame2 = cv.imread('data/phone_5.png')
    frame1 = resize_image(frame1)
    frame2 = resize_image(frame2)
    frame3 = resize_image(frame2.copy())
    T = T.transpose()

    R = R.T
    T = -T
    for i in range(0,600,20):
        pt = [i, 250]
        pt = np.append(pt,0.01)

        # print(f"uvs1 shape {pt.shape}, {pt}")
        # print(f"R SHAPE is {R.shape}, pt shape {pt.shape} T shape {T.shape}")
        # print(f"R dot pt {R.dot(pt)}")
        # print(f"From uvs 1 to uvs2")
        pt_new = (np.dot(R, pt)+T.transpose())[0][0:2]
        # print(f"pt new {pt_new} shape {pt_new.shape}")
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        cv.circle(frame1, (int(pt[0]), int(pt[1])), 3, color, -1)
        cv.circle(frame3, (int(pt[0]), int(pt[1])), 3, color, -1)
        cv.circle(frame2, (int(pt_new[0]), int(pt_new[1])), 3, color, -1)

    final_image = concat_images([frame1, frame2,frame3], ['Frame1', 'Frame2','Frame3'])
    cv.namedWindow("BothImages")
    while True:
        cv.imshow('BothImages', final_image)
        key = cv.waitKey(0)
        if key == ord('q'):
            # Quit the program if 'q' key is pressed
            break



mtx1, dist1 = calibrate_camera(images_folder='D2/*')
mtx2, dist2 = calibrate_camera(images_folder='J2/*')

R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, 'data/*')

# this call might cause segmentation fault error. This is due to calling cv.imshow() and plt.show()
triangulate(mtx1, mtx2, R, T)