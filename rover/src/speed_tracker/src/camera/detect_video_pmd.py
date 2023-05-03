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
ARM = True

import argparse
import math

import queue
import sys
import threading
import rospy

if not ARM:
    import camera.pmd_camera_utils.roypy as roypy
    from camera.pmd_camera_utils.sample_camera_info import print_camera_info
    from camera.pmd_camera_utils.roypy_sample_utils import CameraOpener, add_camera_opener_options
    from camera.pmd_camera_utils.roypy_platform_utils import PlatformHelper
else:
    import camera.pmd_camera_utils_pi.roypy as roypy
    from camera.pmd_camera_utils_pi.sample_camera_info import print_camera_info
    from camera.pmd_camera_utils_pi.roypy_sample_utils import CameraOpener, add_camera_opener_options
    from camera.pmd_camera_utils_pi.roypy_platform_utils import PlatformHelper

import time
import numpy as np
import cv2
from camera.utils.dir_utils import get_next_number
from std_msgs.msg import Float64MultiArray


def clamp_values(values, maximum_value):
    """
    Adjusts the z value based on some criteria.
    """
    clamped_values = np.minimum(maximum_value, values)
    new_values = clamped_values / maximum_value * 255
    return new_values


def key_listener(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Left button pressed")
        gray = param['gray']
        depth = param['depth']
        stack = param['stack']
        print(f"Depth shape {depth.shape}, gray shape {gray.shape}, stack shape {stack.shape}")
        next_number = get_next_number('./data')
        cv2.imwrite(f'./data/depth_{next_number}.png', depth)
        cv2.imwrite(f'./data/gray_{next_number}.png', gray)
        print(f"Data saved #{next_number}")


class MyListener(roypy.IDepthDataListener):
    def __init__(self, q):
        super(MyListener, self).__init__()
        self.subscriber = rospy.Subscriber('/robo_cop/yolov5/bounding_box', Float64MultiArray, self.callback)
        self.publisher = rospy.Publisher('/robo_cop/pmd/distance_deviation', Float64MultiArray)
        self.frame = 0
        self.done = False
        self.undistort_image = False
        self.lock = threading.Lock()
        self.once = False
        self.queue = q
        self.bbox_pts = []
        self.frame_count = 0  # to count total frames
        self.total_fps = 0  # to get the final frames per second
        self.CONFIDENCE_THRESHOLD = 50

    def callback(self, data: Float64MultiArray):
        self.bbox_pts = data.data

    def onNewData(self, data):
        p = data.npoints()
        self.queue.put(p)

    def paint(self, data):
        """Called in the main thread, with data containing one of the items that was added to the
        queue in onNewData.
        """
        # mutex to lock out changes to the distortion while drawing
        self.lock.acquire()
        start_time = time.time()

        # Get PMD Values
        x = data[:, :, 1]
        depth = data[:, :, 2]
        gray = data[:, :, 4]
        max_val = np.max(gray)
        min_val = np.min(gray)
        confidence = data[:, :, 5]

        # Pre-process Depth and Gray values
        x_image = np.zeros(x.shape, np.float32)
        z_image = np.zeros(depth.shape, np.float32)
        gray_image = np.zeros(depth.shape, np.float32)
        mask = confidence > self.CONFIDENCE_THRESHOLD
        x_image[mask] = clamp_values(x_image[mask], maximum_value=5)
        z_image[mask] = clamp_values(depth[mask], maximum_value=5)
        gray_image[mask] = clamp_values(gray[mask], maximum_value=180)
        x_image8 = np.uint8(x_image)
        z_image8 = np.uint8(z_image)
        gray_image8 = np.uint8(gray_image)
        # Apply undistortion
        if self.undistort_image:
            z_image8 = cv2.undistort(z_image8, self.cameraMatrix, self.distortionCoefficients)
            gray_image8 = cv2.undistort(gray_image8, self.cameraMatrix, self.distortionCoefficients)
        mask = confidence < self.CONFIDENCE_THRESHOLD
        depth[mask] = None
        x[mask] = None
        mask = depth == 0
        depth[mask] = None
        mask = x == 0
        x[mask] = None


        depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_LINEAR)
        depth = cv2.rotate(depth, cv2.ROTATE_90_COUNTERCLOCKWISE)

        x = cv2.resize(x, (640, 480), interpolation=cv2.INTER_LINEAR)
        x = cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE)


        z_image8 = cv2.resize(z_image8, (640, 480), interpolation=cv2.INTER_LINEAR)
        z_image8 = cv2.rotate(z_image8, cv2.ROTATE_90_COUNTERCLOCKWISE)
        z_image8 = cv2.cvtColor(z_image8, cv2.COLOR_GRAY2BGR)


        if len(self.bbox_pts) > 0:
            distance = -1
            deviation = -1
            # Point from YOLO
            x1, y1 = self.bbox_pts[0],self.bbox_pts[1]
            x2, y2 = self.bbox_pts[2],self.bbox_pts[3]
            pt1 = (x1,y1)
            pt2 = (x2,y2)

            diff = abs(pt1[0] - pt2[0])
            middle_yolo = (int((pt1[0] + pt2[0]) / 2), int(pt1[1] + diff * 0.1))

            # Point to use
            x1_pico = x1
            y1_pico = y1
            x2_pico = x2
            y2_pico = y2

            w = 480
            # Calculate the percentage of how far the point is from the left edge
            distance_from_left = middle_yolo[0]
            percentage_from_left = distance_from_left / w * 100
            max_diff_x = 491 - 398
            max_diff_y = 294 - 273
            # print(f"Old x1 x2 {x1_pico} {x2_pico}")

            y_diff = max_diff_y
            y1_pico -= y_diff
            if y1_pico < 0:
                y1_pico = 0
            y2_pico -= y_diff
            if y2_pico < 0:
                y2_pico = 0
            if percentage_from_left > 35:
                # Define the new range
                new_min = 0
                new_max = max_diff_x

                # Map the original values to the new range
                x_diff = np.interp(percentage_from_left, (35, 100), (new_min, new_max))

                # print(f"percentage_from_left {percentage_from_left} x diff {x_diff}")
                x1_pico += x_diff
                if x1_pico > w:
                    x1_pico = w
                x2_pico += x_diff
                if x2_pico > w:
                    x2_pico = w

                # print(f"New x1 x2 {x1_pico} {x2_pico}")

            # print(f"Percentage from left {percentage_from_left}")

            x1_pico = int(x1_pico)
            y1_pico = int(y1_pico)
            x2_pico = int(x2_pico)
            y2_pico = int(y2_pico)
            pt1_pico = (int(x1_pico), int(y1_pico))
            pt2_pico = (int(x2_pico), int(y2_pico))

            depth_bbox = depth[y1_pico:y2_pico, x1_pico:x2_pico]

            center = ((pt1_pico[0] + pt2_pico[0]) / 2, (pt1_pico[1] + pt2_pico[1]) / 2)
            width = pt2_pico[0] - pt1_pico[0]
            height = pt2_pico[1] - pt1_pico[1]
            new_width = int(width * 0.6)
            new_pt1_pico = (int(center[0] - new_width / 2), pt1_pico[1])
            new_pt2_pico = (int(center[0] + new_width / 2), pt2_pico[1])
            x_bbox = x[new_pt1_pico[1]:new_pt2_pico[1], new_pt1_pico[0]:new_pt2_pico[0]]
            color = (125, 125, 0)
            cv2.rectangle(z_image8, new_pt1_pico, new_pt2_pico, color, 2)

            # print(f"Max depth {np.max(depth_bbox)} {np.min(depth_bbox)}")
            diff = abs(pt1_pico[0] - pt2_pico[0])
            middle = (int((pt1_pico[0] + pt2_pico[0]) / 2), int(pt1_pico[1] + diff * 0.1))
            color = (255, 255, 0)

            cv2.circle(z_image8, (int(middle[0]), int(middle[1])), 5, color, -1)

            # full_0_count = np.count_nonzero(z_image8_bbox == 0)
            hist, bins = np.histogram(depth_bbox, bins=np.linspace(0, 5, 15))
            # print(f"HIST ARE {hist}")
            # print(hist)
            # Find bin with most pixels
            max_bin = np.argmax(hist)
            min_value_pixel = bins[max_bin]
            max_value_pixel = bins[max_bin + 1]
            distance_count = depth_bbox[(depth_bbox >= min_value_pixel) & (depth_bbox <= max_value_pixel)]
            if len(distance_count) > 0:
                distance = np.mean(distance_count)
            # print(f"We got the distance {distance} from the count {distance_count}")
            hist, bins = np.histogram(x_bbox, bins=np.linspace(-1, 1, 15))
            # Find bin with most pixels
            max_bin = np.argmax(hist)
            min_value_pixel = bins[max_bin]
            max_value_pixel = bins[max_bin + 1]
            x_count = x_bbox[(x_bbox >= min_value_pixel) & (x_bbox <= max_value_pixel)]
            if len(x_count) > 0:
                x_mean = np.mean(x_count)
                deviation = math.degrees(math.atan2(x_mean, distance))
            # print(f"We got the deviation {deviation} from the count {x_count}")
            if distance is not None and deviation is not None:
                # print(f"Distance is {distance} deviation is {deviation}")
                distance_dev_arr = Float64MultiArray()
                distance_dev_arr.data = [distance, deviation]
                self.publisher.publish(distance_dev_arr)



        end_time = time.time()
        # get the fps
        fps = 1 / (end_time - start_time)
        # add fps to total fps
        self.total_fps += fps
        # increment frame count
        self.frame_count += 1
        # write the FPS on the current frame
        # cv2.putText(gray_image8, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
        #             1, (0, 255, 0), 2)
        # convert from BGR to RGB color format
        # image = cv2.cvtColor(gray_image8, cv2.COLOR_BGR2RGB)
        # cv2.imshow('image', image)

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
        self.undistort_image = not self.undistort_image
        self.lock.release()

    # Map the depth values from the camera to 0..255


def main():
    rospy.init_node("pmd_camera")
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

    cam.setUseCase("MODE_9_10FPS_1000")
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
    while not rospy.is_shutdown():
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
