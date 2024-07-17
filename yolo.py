from collections import defaultdict

import cv2
import numpy as np
import copy
import json
import argparse
import torch
import os

from ultralytics import YOLO

def IoA(box1, box2, box_format="xyxy"):
    """
    Calculate the Intersection over Area (IoA) of two bounding boxes.

    Parameters:
        box1 (list): first box, list of four coordinates (x1, y1, x2, y2)
        box2 (list): second box, list of four coordinates (x1, y1, x2, y2)
        box_format (str): format of the boxes, 'xyxy' or 'xywh'

    Returns:
        float: IoA value
    """

    if box_format == "xywh":
        # Convert the boxes to the format (x1, y1, x2, y2)
        box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
        box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    # Calculate the intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the area of the box2
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the IoA
    ioa = intersection / area_box2

    return ioa

def IoU(box1, box2, box_format="xyxy"):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
        box1 (list): first box, list of four coordinates (x1, y1, x2, y2)
        box2 (list): second box, list of four coordinates (x1, y1, x2, y2)
        box_format (str): format of the boxes, 'xyxy' or 'xywh'

    Returns:
        float: IoU value
    """

    if box_format == "xywh":
        # Convert the boxes to the format (x1, y1, x2, y2)
        box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
        box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    # Calculate the intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the area of the two boxes
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the union area
    union = area_box1 + area_box2 - intersection

    # Calculate the IoU
    iou = intersection / union

    return iou 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PrivacyFilter')
    parser.add_argument('-ivp', '--input-video-path', default='default.mp4', help='input video path')
    parser.add_argument('-of', '--output-folder', default='outputs', help='output folder')
    parser.add_argument('-w', '--weight', default='yolov8s-world.pt', help='weight path')
    # parser.add_argument()

    args = parser.parse_args()

    classes_name = ["ID card", "paper", "house plate number"]
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Load the YOLOv8 model
    model = YOLO(args.weight, device=device)
    model.set_classes(classes_name)

    # Open the video file
    video_path = args.input_video_path
    cap = cv2.VideoCapture(video_path)

    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
    fps = cap.get(cv2.CAP_PROP_FPS)

    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    file_name = video_path.split('/')[-1].split('.')[0]

    out_path = os.path.join(args.output_folder, file_name+"_detect")
    out_path_blur = os.path.join(args.output_folder, file_name+"_blur")
    writer = cv2.VideoWriter(out_path + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    writer_blur = cv2.VideoWriter(out_path_blur + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    track_dict = dict()
    track_list = []

    frames = []
    frame_list = []
    frame_blur_list = []

    max_absent = 20

    colors = [
            [255, 127, 0], [127, 255, 0], [0, 255, 127], [0, 127, 255], [127, 0, 255], [255, 0, 127],
            [255, 255, 255],
            [127, 0, 127], [0, 127, 127], [127, 127, 0], [127, 0, 0], [127, 0, 0], [0, 127, 0],
            [127, 127, 127],
            [255, 0, 255], [0, 255, 255], [255, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0],
            [0, 0, 0],
            [255, 127, 255], [127, 255, 255], [255, 255, 127], [127, 127, 255], [255, 127, 127], [255, 127, 127],
        ]  # 27 colors

    frame_id = -1
    min_first_confidence = 0.03

    object_id = -1

    frame_area = h * w

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            blur_frame = frame.copy()
            original_frame = frame.copy()
            frame_id += 1
            results = model.predict(frame, conf=0.01, save=False)

            boxes = results[0].boxes
            boxes_xyxy = boxes.xyxy
            confs = boxes.conf
            classes = boxes.cls
            if boxes_xyxy.shape[0]:
                for i in range(boxes_xyxy.shape[0]):
                    box = boxes_xyxy[i].cpu().tolist()
                    conf = confs[i].item()
                    cls = int(classes[i].item())

                    box_area = (box[2] - box[0]) * (box[3] - box[1])
                    if box_area > 0.3 * frame_area:
                        continue

                    is_tracked = False

                    for key, value in track_dict.items():
                        if  IoU(value['box'], box) > 0.5 and not value['ignore'] and not is_tracked:
                            if cls != value['cls']:
                                if value['current_frame_id'] - value['start_frame_id'] >= 5:
                                    cls = value['cls']
                                else:
                                    continue

                            value['conf'] = conf
                            value['box'] = box
                            value['cls'] = cls
                            value['absent'] = -1
                            value['current_frame_id'] = frame_id
                            object_id_local = int(key)

                            is_tracked = True
                            break

                    if not is_tracked and conf > min_first_confidence:
                        obj = {
                            'conf': conf,
                            'box': box,
                            'cls': cls,
                            'absent': -1,
                            'current_frame_id': frame_id,
                            'start_frame_id': frame_id,
                            'ignore': False
                        }
                        object_id += 1
                        object_id_local = object_id
                        track_dict[str(object_id)] = obj
                        is_tracked = True
                    
                    if is_tracked:
                        color = colors[object_id_local % len(colors)]
                        x1, y1, x2, y2 = box
                        x1 = max(0, int(x1))
                        y1 = max(0, int(y1))
                        x2 = min(w - 1, int(x2))
                        y2 = min(h - 1, int(y2))
                        blur_frame[y1:y2 + 1, x1:x2 + 1] = cv2.blur(blur_frame[y1:y2 + 1, x1:x2+1], (50, 50))

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"ID: {object_id_local}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                for key, value in track_dict.items():
                    if not value['ignore']:
                        value['absent'] += 1

            else:
                for key, value in track_dict.items():
                    if not value['ignore']:
                        value['absent'] += 1

                    if value['absent'] <= max_absent:
                        x1, y1, x2, y2 = value['box']
                        x1 = max(0, int(x1))
                        y1 = max(0, int(y1))
                        x2 = min(w - 1, int(x2))
                        y2 = min(h - 1, int(y2))
                        color = colors[int(key) % len(colors)]
                        blur_frame[y1:y2 + 1, x1:x2 + 1] = cv2.blur(blur_frame[y1:y2 + 1, x1:x2+1], (50, 50))
                        # print('color', color, object_id_local)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        cv2.putText(frame, f"ID: {key}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            for key, value in track_dict.items():
                if value['absent'] > max_absent:
                    value['ignore'] = True
                else:
                    value['current_frame_id'] = frame_id

            frame_blur_list.append(blur_frame)
            frame_list.append(frame)
            frames.append(original_frame)

            writer.write(frame)
            writer_blur.write(blur_frame)

        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    writer.release()
    writer_blur.release()

    for key in track_dict.keys():
        track_dict[key]['start_time'] = track_dict[key]['start_frame_id'] / fps
        track_dict[key]['end_time'] = track_dict[key]['current_frame_id'] / fps
        track_dict[key]['name'] = classes_name[int(track_dict[key]['cls'])]

    json_file = os.path.join(args.output_folder, 'objects_list.json')
    with open(json_file, 'w') as f:
        json.dump(track_dict, f)