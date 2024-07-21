from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request
from fastapi.params import Path
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import cv2
from ultralytics import YOLO
import moviepy.editor as mp
import whisper_timestamped as whisper
import shutil
import requests
from transformers import pipeline
from pydub import AudioSegment
import httpx
import re
import aiofiles


classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
categories = ["Profanity", "Offensive Language", "Inappropriate Content", "Harmful Content", "Leak Info"]


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

def check_word(text):
    results = classifier(text, candidate_labels=categories)
    labels = results['labels']
    scores = results['scores']
    filtered_labels = [label for label, score in zip(labels, scores) if score > 0.6]
    return filtered_labels

def video_to_audio(video_path, audio_path):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

def audio_to_text(audio_path):
    audio = whisper.load_audio(audio_path)
    model = whisper.load_model("base")
    result = whisper.transcribe(model, audio)
    return result["segments"]

def create_beep_sound() -> AudioSegment:
    # Create a beep sound
    beep = AudioSegment.from_file("beep.wav")
    return beep

def insert_beeps(audio_path: str, segments: list):
    beep = create_beep_sound()
    audio = AudioSegment.from_file(audio_path)
    
    for segment in segments:
        for word in segment["words"]:
            if word["filtered"]:
                beep_start_time = word["start"] * 1000  # pydub uses milliseconds
                beep_end_time = word["end"] * 1000
                beep_duration = beep_end_time - beep_start_time

                # Extend the beep duration to fit the word duration
                adjusted_beep = beep[:beep_duration] if len(beep) > beep_duration else beep + AudioSegment.silent(duration=beep_duration - len(beep))
                
                # Insert beep sound
                audio = audio[:int(beep_start_time)] + adjusted_beep + audio[int(beep_end_time):]

    audio.export(audio_path, format="mp3")

@app.post("/get_audio")
async def get_audio(file: UploadFile = File(...)):
    os.makedirs("audio", exist_ok=True)
    os.makedirs("temp", exist_ok=True)

    video_path = f"temp/{file.filename}"
    audio_path = "audio/audio.mp3"
    
    # Save uploaded video file
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Convert video to audio
    video_to_audio(video_path, audio_path)
    
    # Transcribe audio to text
    segments = audio_to_text(audio_path)
    
    os.remove(video_path)
    # os.remove(audio_path)
    
    for segment in segments:
        for word in segment["words"]:
            word["filtered"] = check_word(word["text"])
    
    insert_beeps(audio_path, segments)
    return JSONResponse(segments)


UPLOAD_API_URL = "https://files.dev.tekoapis.net/upload/video/"  # Replace with the actual target API URL
DOWNLOAD_API_URL = "https://files.dev.tekoapis.net/files/{uuid}"  # Replace with the actual target API URL

@app.post("/get_oauth_token/")
async def get_oauth_token():
    url = 'https://oauth.dev.tekoapis.net/oauth/token'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    data = {
        'grant_type': 'password',
        'scope': 'openid profile us voucher-hub read:permissions tenant:management',
        'client_id': '5da48d702722494b9ff1a792137eb8a6',
        'username': '0949590040',
        'password': '12345678'
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, data=data)
        if response.status_code == 200:
            response_data = response.json()
            return JSONResponse(content=response_data)
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)

@app.get("/get_access_token/")
async def get_access_token():
    url = 'http://localhost:8000/get_oauth_token/'

    async with httpx.AsyncClient() as client:
        response = await client.post(url)
        if response.status_code == 200:
            response_data = response.json()
            access_token = response_data.get("access_token")
            print(access_token)
            if access_token:
                return access_token
            else:
                raise HTTPException(status_code=500, detail="Access token not found in the response")
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    access_token = await get_access_token()
    url = 'https://files.dev.tekoapis.net/upload/video'
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }

    form_data = {
        'file': (file.filename, await file.read(), file.content_type),
        'cloud': (None, 'false')
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, files=form_data)
        if response.status_code == 200:
            response_data = response.json()
            video_url = response_data.get("url")
            if video_url:
                # Extract UUID from the video URL using regex
                match = re.search(r'/([0-9a-fA-F-]{36})/', video_url)
                if match:
                    uuid = match.group(1)
                    with open("uuid.txt", "w") as uuid_file:
                        uuid_file.write(uuid)
                    return JSONResponse(content={"uuid": uuid, "url": video_url})
                else:
                    raise HTTPException(status_code=500, detail="UUID not found in the video URL")
            else:
                raise HTTPException(status_code=500, detail="URL not found in the response")
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)

@app.get("/get_uuid/")
async def get_uuid_from_file():
    try:
        with open("uuid.txt", "r") as uuid_file:
            uuid = uuid_file.read().strip()
            return {"uuid": uuid}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="UUID file not found")

@app.get("/download_video/")
async def download_video():
    uuid_response = await get_uuid_from_file()
    uuid = uuid_response["uuid"]
    url = f'https://files.dev.tekoapis.net/files/{uuid}'
    headers = {
        'Accept': 'application/binary'
    }

    video_file_path = f"/tmp/{uuid}.mp4"

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        if response.status_code == 200:
            async with aiofiles.open(video_file_path, 'wb') as video_file:
                await video_file.write(response.content)
            return FileResponse(video_file_path, media_type='video/mp4', filename=f"{uuid}.mp4")
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
