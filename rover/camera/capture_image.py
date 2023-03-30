#!/usr/bin/python3

# Copyright (C) 2019 Infineon Technologies & pmdtechnologies ag
#
# THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY
# KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A
# PARTICULAR PURPOSE.

"""This sample shows how to use openCV on the depthdata we get back from either a camera or an rrf file.
The Camera's lens parameters are optionally used to remove the lens distortion and then the image is displayed using openCV windows.
Press 'd' on the keyboard to toggle the distortion while a window is selected. Press esc to exit.
"""

import argparse
import pmd_camera_utils.roypy as roypy
import queue
import sys
import threading
from pmd_camera_utils.sample_camera_info import print_camera_info
from pmd_camera_utils.roypy_sample_utils import CameraOpener, add_camera_opener_options
from pmd_camera_utils.roypy_platform_utils import PlatformHelper

import numpy as np
import cv2


def adjustZValue(zValues):
    """
    Adjusts the z value based on some criteria.
    """
    clampedDist = np.minimum(5, zValues)
    newZValues = clampedDist / 5 * 255
    return newZValues


# Map the gray values from the camera to 0..255
def adjustGrayValue(grayValues):
    """
    Adjusts the gray value based on some criteria.
    """
    clampedVal = np.minimum(800, grayValues)
    newGrayValues = clampedVal / 800 * 255
    return newGrayValues


def key_listener(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Left button pressed")

        data = param['data']
        print(f"Data shape before save {data.shape}")
        cv2.imwrite('output.jpg', data)
        print(f"Data saved")

class MyListener(roypy.IDepthDataListener):
    def __init__(self, q):
        super(MyListener, self).__init__()
        self.frame = 0
        self.done = False
        self.undistortImage = False
        self.lock = threading.Lock()
        self.once = False
        self.queue = q

    def onNewData(self, data):
        p = data.npoints()
        self.queue.put(p)

    def paint(self, data):
        """Called in the main thread, with data containing one of the items that was added to the
        queue in onNewData.
        """
        # mutex to lock out changes to the distortion while drawing
        self.lock.acquire()

        depth = data[:, :, 2]
        gray = data[:, :, 4]
        confidence = data[:, :, 5]

        zImage = np.zeros(depth.shape, np.float32)
        grayImage = np.zeros(depth.shape, np.float32)

        # Pre-process Depth and Gray values
        mask = confidence > 0
        zImage[mask] = adjustZValue(depth[mask])
        grayImage[mask] = adjustGrayValue(gray[mask])

        zImage8 = np.uint8(zImage)
        grayImage8 = np.uint8(grayImage)

        # apply undistortion
        if self.undistortImage:
            zImage8 = cv2.undistort(zImage8, self.cameraMatrix, self.distortionCoefficients)
            grayImage8 = cv2.undistort(grayImage8, self.cameraMatrix, self.distortionCoefficients)

        # finally show the images
        cv2.namedWindow("Depth")
        cv2.imshow('Depth', zImage8)
        cv2.imshow('Gray', grayImage8)
        cv2.setMouseCallback("Depth", key_listener, {'data': grayImage8})

        self.lock.release()
        self.done = True

    def setLensParameters(self, lensParameters):
        # Construct the camera matrix
        # (fx   0    cx)
        # (0    fy   cy)
        # (0    0    1 )
        self.cameraMatrix = np.zeros((3, 3), np.float32)
        self.cameraMatrix[0, 0] = lensParameters['fx']
        self.cameraMatrix[0, 2] = lensParameters['cx']
        self.cameraMatrix[1, 1] = lensParameters['fy']
        self.cameraMatrix[1, 2] = lensParameters['cy']
        self.cameraMatrix[2, 2] = 1

        # Construct the distortion coefficients
        # k1 k2 p1 p2 k3
        self.distortionCoefficients = np.zeros((1, 5), np.float32)
        self.distortionCoefficients[0, 0] = lensParameters['k1']
        self.distortionCoefficients[0, 1] = lensParameters['k2']
        self.distortionCoefficients[0, 2] = lensParameters['p1']
        self.distortionCoefficients[0, 3] = lensParameters['p2']
        self.distortionCoefficients[0, 4] = lensParameters['k3']

    def toggleUndistort(self):
        self.lock.acquire()
        self.undistortImage = not self.undistortImage
        self.lock.release()

    # Map the depth values from the camera to 0..255


def main():
    # Set the available arguments
    platformhelper = PlatformHelper()
    parser = argparse.ArgumentParser(usage=__doc__)
    add_camera_opener_options(parser)
    options = parser.parse_args()

    opener = CameraOpener(options)

    try:
        cam = opener.open_camera()
    except:
        print("could not open Camera Interface")
        sys.exit(1)

    try:
        # retrieve the interface that is available for recordings
        replay = cam.asReplay()
        print("Using a recording")
        print("Framecount : ", replay.frameCount())
        print("File version : ", replay.getFileVersion())
    except SystemError:
        print("Using a live camera")

    cam.setUseCase("MODE_5_35FPS_600")
    q = queue.Queue()
    l = MyListener(q)
    cam.registerDataListener(l)
    cam.startCapture()

    lensP = cam.getLensParameters()
    l.setLensParameters(lensP)

    process_event_queue(q, l)

    cam.stopCapture()
    print("Done")


def process_event_queue(q, painter):
    while True:
        try:
            # try to retrieve an item from the queue.
            # this will block until an item can be retrieved
            # or the timeout of 1 second is hit
            if len(q.queue) == 0:
                item = q.get(True, 1)
            else:
                for i in range(0, len(q.queue)):
                    item = q.get(True, 1)
        except queue.Empty:
            # this will be thrown when the timeout is hit
            break
        else:
            painter.paint(item)
            # waitKey is required to use imshow, we wait for 1 millisecond
            currentKey = cv2.waitKey(1)
            if currentKey == ord('d'):
                painter.toggleUndistort()
            # close if escape key pressed
            if currentKey == 27:
                break


if __name__ == "__main__":
    main()