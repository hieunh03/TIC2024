from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import json
from collections import defaultdict
import moviepy.editor as mp
import whisper_timestamped as whisper

app = FastAPI()

# Allow CORS for frontend interaction if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def IoA(box1, box2, box_format="xyxy"):
    if box_format == "xywh":
        box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
        box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    ioa = intersection / area_box2

    return ioa

def IoU(box1, box2, box_format="xyxy"):
    if box_format == "xywh":
        box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
        box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_box1 + area_box2 - intersection
    iou = intersection / union

    return iou

@app.post("/process_video/")
async def upload_video(file: UploadFile = File(...), weight: str = 'yolov8s-worldv2.pt', output_folder: str = 'outputs'):
    video_path = f"uploaded_{file.filename}"
    
    with open(video_path, "wb") as video_file:
        video_file.write(file.file.read())

    classes_name = ["ID card", "Paper"]

    model = YOLO(weight)
    model.set_classes(classes_name)

    cap = cv2.VideoCapture(video_path)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_name = video_path.split('/')[-1].split('.')[0]
    out_path = os.path.join(output_folder, file_name+"_detect")
    out_path_blur = os.path.join(output_folder, file_name+"_blur")
    writer = cv2.VideoWriter(out_path + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    writer_blur = cv2.VideoWriter(out_path_blur + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    track_dict = dict()
    max_absent = 20
    colors = [
        [255, 127, 0], [127, 255, 0], [0, 255, 127], [0, 127, 255], [127, 0, 255], [255, 0, 127],
        [255, 255, 255], [127, 0, 127], [0, 127, 127], [127, 127, 0], [127, 0, 0], [127, 0, 0], [0, 127, 0],
        [127, 127, 127], [255, 0, 255], [0, 255, 255], [255, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0],
        [0, 0, 0], [255, 127, 255], [127, 255, 255], [255, 255, 127], [127, 127, 255], [255, 127, 127], [255, 127, 127]
    ]

    frame_id = -1
    min_first_confidence = 0.03
    object_id = -1
    frame_area = h * w

    while cap.isOpened():
        success, frame = cap.read()
        if success:
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
                        if IoU(value['box'], box) > 0.5 and not value['ignore'] and not is_tracked:
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
                        blur_frame[y1:y2 + 1, x1:x2 + 1] = cv2.blur(blur_frame[y1:y2 + 1, x1:x2 + 1], (50, 50))
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
                        blur_frame[y1:y2 + 1, x1:x2 + 1] = cv2.blur(blur_frame[y1:y2 + 1, x1:x2 + 1], (50, 50))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"ID: {key}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            for key, value in track_dict.items():
                if value['absent'] > max_absent:
                    value['ignore'] = True
                else:
                    value['current_frame_id'] = frame_id
            writer.write(frame)
            writer_blur.write(blur_frame)
        else:
            break

    cap.release()
    writer.release()
    writer_blur.release()

    for key in track_dict.keys():
        track_dict[key]['start_time'] = track_dict[key]['start_frame_id'] / fps
        track_dict[key]['end_time'] = track_dict[key]['current_frame_id'] / fps
        track_dict[key]['name'] = classes_name[int(track_dict[key]['cls'])]

    return JSONResponse({"detection_video": out_path + '.mp4', "blurred_video": out_path_blur + '.mp4', "data": track_dict})

def video_to_audio(video_path, audio_path):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

def audio_to_text(audio_path):
    audio = whisper.load_audio(audio_path)
    model = whisper.load_model("base")
    result = whisper.transcribe(model, audio)
    return result

@app.get("/get_audio")
async def get_audio():
    video_to_audio('video.mp4', 'audio/audio.mp3')
    stt_result = audio_to_text('audio/audio.mp3')
    return stt_result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
