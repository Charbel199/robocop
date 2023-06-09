#! /usr/bin/env python
import cv2
import rospy
import torch
import argparse
import time
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from std_msgs.msg import Float64MultiArray
from ocr import ocr_plate
from mongo_db import get_name_from_license_plate

def load_model(weights_path,
               device,
               half=False):
    model = attempt_load(weights_path, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    return model, stride, names


def detect_objects(model,
                   img,
                   draw_image,
                   augment=False,
                   visualize=False,
                   conf_thresh=0.25,
                   iou_thresh=0.45,
                   agnostic_nms=False,
                   classes=None,
                   max_det=1000,
                   line_thickness=3
                   ):
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz = check_img_size([480, 640], floor=gs * 2)  # verify imgsz is gs-multiple

    pred = model(img, augment=augment, visualize=visualize)[0]
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    # NMS
    pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes, agnostic_nms, max_det=max_det)

    annotator = Annotator(draw_image, line_width=line_thickness, pil=not ascii)

    car_detections = []

    # Process predictions
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], draw_image.shape).round()
            hide_labels = False
            hide_conf = False
            # Write results
            for *xyxy, conf, cls in reversed(det):
                p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}')
                conf_label = f'{conf:.2f}'
                if 'rover' in label:
                    car_detections.append({
                        'p1': p1,
                        'p2': p2,
                        'label': label,
                        'conf': conf_label
                    })
                annotator.box_label(xyxy, f"{label} {conf_label}", color=colors(c, True))

    return annotator.result(), car_detections

def send_tts(text, similarity, stability):
    import requests

    url = 'http://localhost:6300/tts'

    params = {
        'text': text,
        'stability': stability,
        'similarity': similarity
    }

    headers = {
        'accept': 'application/json'
    }

    response = requests.post(url, params=params, headers=headers)

    return response.content
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/train/exp17/weights/best.pt', help='Initial weights path')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--half', type=bool, default=False, help='Half precision')

    opt = parser.parse_args()
    return opt


class Yolov5ROS:
    def __init__(self):
        self.publisher = rospy.Publisher('/robo_cop/yolov5/bounding_box', Float64MultiArray, queue_size=10)
        self.subscriber = rospy.Subscriber('/robo_cop/pmd/distance_deviation', Float64MultiArray, self.callback)
        self.distance = 5

    def callback(self, data: Float64MultiArray):
        self.distance = data.data[0]


def main(opt):
    yolov5_ros = Yolov5ROS()
    yolov5_publisher = yolov5_ros.publisher

    # define the computation device
    device = opt.device
    device = select_device(device)
    half = opt.half
    half &= device.type != 'cpu'
    # Load model
    # weights='yolov5m.pt'
    weights = opt.weights
    model, stride, names = load_model(weights_path=weights,
                                      device=device,
                                      half=False)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    frame_count = 0  # to count total frames
    total_fps = 0  # to get the final frames per second
    last_time = None
    plate_found = False
    # read until end of video
    while (cap.isOpened()):
        ret, img = cap.read()
        try:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # img = cv2.imread('test_images/imtest1.jpeg')
            # img = cv2.resize(img, (448,672))
            original_image_copy = img.copy()

            if ret:
                # get the start time
                start_time = time.time()

                img = torch.from_numpy(img).to(device)
                img = torch.movedim(img, 2, 0)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img = img / 255.0  # 0 - 255 to 0.0 - 1.0
                if len(img.shape) == 3:
                    img = img[None]  # expand for batch dim

                img_view, car_detections = detect_objects(model=model,
                                                          img=img,
                                                          augment=False,
                                                          visualize=False,
                                                          draw_image=original_image_copy.copy())
                bbox_region = []
                if len(car_detections) > 0:
                    highest_car_confidence = max(car_detections, key=lambda x: x['conf'])
                    bbox_data = Float64MultiArray()
                    x1 = highest_car_confidence['p1'][0]
                    y1 = highest_car_confidence['p1'][1]
                    x2 = highest_car_confidence['p2'][0]
                    y2 = highest_car_confidence['p2'][1]
                    bbox_data.data = [x1, y1, x2, y2]
                    yolov5_publisher.publish(bbox_data)

                    if not plate_found and yolov5_ros.distance < 0.3:
                    # if not plate_found and True:
                        current_time = time.time()
                        if last_time is None or (current_time - last_time) >= 10:
                            bbox_region = original_image_copy[y1:y2, x1:x2]
                            plate_number, confidence = ocr_plate(bbox_region.copy())
                            print(f"Plate number is {plate_number} with confidence {confidence}")

                            ## CHECK IF IN DB
                            # IF IN DB
                            name_from_license_plate = get_name_from_license_plate(plate_number)
                            print(f"Name from license plate {name_from_license_plate}")
                            if name_from_license_plate is not None:
                                plate_found = True
                                # RUN TTS
                                response = send_tts(f"{name_from_license_plate} STOP RIGHT THERE !!! THIS IS THE POLICE !!!! PULL OVER IMMEDIATELY !!!!! RATATATAT",
                                                    stability=0.4,
                                                    similarity=1)
                                print(f"Response from tts {response}")
                            last_time = current_time

                else:
                    bbox_data = Float64MultiArray()
                    bbox_data.data = []
                    yolov5_publisher.publish(bbox_data)
                # print(f"Car detections: {car_detections}")
                end_time = time.time()
                fps = 1 / (end_time - start_time)
                total_fps += fps
                frame_count += 1
                cv2.putText(img_view, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

                cv2.imshow('image', img_view)
                # if len(bbox_region)>0:
                #     cv2.imshow('bbox_region', bbox_region)
                # press `q` to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except:
            continue

    # release VideoCapture()
    cap.release()
    # close all frames and video windows
    cv2.destroyAllWindows()

    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")


if __name__ == "__main__":
    opt = parse_args()
    rospy.init_node('yolov5')
    main(opt)
