import asyncio, time
import cv2
import numpy as np
from ultralytics import YOLO
from reolinkapi import Camera
from dotenv import load_dotenv
import os

load_dotenv()
CAM_IP = os.getenv("CAM_IP")
USER = os.getenv("USER")
PASS = os.getenv("PASS")

model = YOLO("yolov8n.pt")  # small, fast

async def pet_in_frame(cam) -> tuple[bool, float]:
    jpg = await cam.get_snapshot()
    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

    results = model.predict(frame, imgsz=640, conf=0.35, verbose=False)[0]
    # COCO class id 16 = dog, 15 = cat
    dog_confs = [float(b.conf) for b in results.boxes if int(b.cls) == 16]

    if dog_confs:
        return True, max(dog_confs)
    return False, 0.0

async def scan_for_dog():
    cam = Camera(CAM_IP, USER, PASS)
    await cam.connect()

    found = False
    while not found:
        # small pan step to the right
        await cam.ptz_control(command="Right", speed=20)
        await asyncio.sleep(0.4)
        await cam.ptz_control(command="Stop")

        found, conf = await pet_in_frame(cam)
        print("dog?", found, "conf", conf)

        if found:
            # zoom in a bit and stop scanning
            await cam.ptz_control(command="ZoomIn", speed=10)
            await asyncio.sleep(0.6)
            await cam.ptz_control(command="Stop")

    await cam.disconnect()

asyncio.run(scan_for_dog())
