import cv2
import time
import argparse
import numpy as np


def yolov3(yolo_weights, yolo_cfg, coco_names):
    '''
    Takes in YOLOv3 files and uses OpenCV to generate network.

        Parameters:
                yolo_weights: weights file for YOLO model
                yolo_cfg: config file for YOLO model
                coco_names: file containing list of classes
    '''
    net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
    classes = open(coco_names).read().strip().split("\n")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return net, classes, output_layers


def perform_detection(net, img, output_layers, w, h, confidence_threshold):
    '''
    Uses YOLO network to perform detection.

        Parameters:
            net: Network created from OpenCV
            img: image to perform detections on
            output_layers: part of the YOLO network, used during forward propagation
            w: width of frame
            h: height of frame
            confidence_threshold: confidence threshold for objects identified to be displayed
    '''
    blob = cv2.dnn.blobFromImage(img, 1 / 255., (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Object is deemed to be detected
            if confidence > confidence_threshold:
                # center_x, center_y, width, height = (detection[0:4] * np.array([w, h, w, h])).astype('int')
                center_x, center_y, width, height = list(map(int, detection[0:4] * [w, h, w, h]))
                # print(center_x, center_y, width, height)

                top_left_x = int(center_x - (width / 2))
                top_left_y = int(center_y - (height / 2))
                bottom_right_x = int(center_x + (width / 2))
                bottom_right_y = int(center_y + (height / 2))

                boxes.append([top_left_x, top_left_y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences, class_ids


def dectection_video_file(webcam, video_path, yolo_weights, yolo_cfg, coco_names, confidence_threshold, nms_threshold):
    '''
    Main function that runs detector and then creates trackers for each object in the frame.

        Parameters:
            webcam: determines whether or not the program uses a live webcam feed
            video_path: filepath of video to be processed
            yolo_weights: weights file for YOLO model
            yolo_cfg: config file for YOLO model
            coco_names: file containing list of classes
            confidence_threshold: confidence threshold for objects identified to be displayed
            nms_threshold: non-max suppression threshold for objects indentified
    '''
    # TODO Try different pre-processing methods; move classifier code to its own function

    net, classes, output_layers = yolov3(yolo_weights, yolo_cfg, coco_names)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    detection = False
    trackers = cv2.MultiTracker_create()

    if webcam:
        video = cv2.VideoCapture(0)
    else:
        video = cv2.VideoCapture(video_path)

    backSub = cv.createBackgroundSubtractorMOG2(varThreshold=10.0)
    while True:
        ret, image = video.read()
        # pre-processing
        if detection:
            image = cv2.GaussianBlur(image, (5, 5), 0)  # Blur
            imageTop = image[0:415, 0:1280]
            imageBot = image[415:720, 0:1280]
            fgMask = backSub.apply(imageTop)
            imageTop = cv2.bitwise_and(imageTop, imageTop, mask=fgMask)
            image = cv2.vconcat([imageTop, imageBot])

        h, w = image.shape[:2]
        if not detection:
            # Create tracker objects from bounding boxes detected by YOLO
            boxes, confidences, class_ids = perform_detection(net, image, output_layers, w, h, confidence_threshold)
            for box in boxes:
                tracker = cv2.TrackerCSRT_create()
                trackers.add(tracker, image, tuple(box))
            detection = True
        (success, boxes_tracker) = trackers.update(image)

        # Draw tracker rectangles
        for box in boxes_tracker:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Write frame to output video and show live output
        result.write(image)
        cv2.imshow("Detection", image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
    result.release()
    video.release()


if __name__ == '__main__':

    ## Arguments to give before running
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', help='Path to video file', default=None)
    ap.add_argument('--camera', help='To use the live feed from web-cam', type=bool, default=False)
    ap.add_argument('--weights', help='Path to model weights', type=str, default='yolov3.weights')
    ap.add_argument('--configs', help='Path to model configs', type=str, default='yolov3.cfg')
    ap.add_argument('--class_names', help='Path to class-names text file', type=str, default='coco.names')
    ap.add_argument('--conf_thresh', help='Confidence threshold value', default=0.5)
    ap.add_argument('--nms_thresh', help='Confidence threshold value', default=0.4)
    args = vars(ap.parse_args())

    # Extract video name from given filepath. Assumes input videos are in "input_videos" directory
    video_name = args['video'][(args['video'].rfind('/') + 1):args['video'].rfind('.')]

    # VideoWriter object that writes modified frames to output video
    result = cv2.VideoWriter('output_videos/result_' + video_name + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10,
                             (1280, 720))

    yolo_weights, yolo_cfg, coco_names = args['weights'], args['configs'], args['class_names']
    confidence_threshold = args['conf_thresh']
    nms_threshold = args['nms_thresh']

    if args['camera'] == True or args['video']:
        webcam = args['camera']
        video_path = args['video']
        dectection_video_file(webcam, video_path, yolo_weights, yolo_cfg, coco_names, confidence_threshold,
                              nms_threshold)
