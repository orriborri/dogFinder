import asyncio
import threading
from flask import Flask, Response, jsonify, render_template_string, request

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
CHANNEL = 0

# Load YOLO once
yolo = YOLO("yolov8n.pt")  # nano = fast

app = Flask(__name__)

# Keep one shared camera connection in a background loop
cam = None
loop = asyncio.new_event_loop()

def start_loop():
    asyncio.set_event_loop(loop)
    loop.run_forever()

threading.Thread(target=start_loop, daemon=True).start()

async def get_cam():
    global cam
    if cam is None:
        cam = Camera(CAM_IP, USER, PASS)
        await cam.connect()
    return cam

def run_async(coro):
    """Run async reolink calls from sync Flask routes."""
    fut = asyncio.run_coroutine_threadsafe(coro, loop)
    return fut.result()

@app.get("/snapshot.jpg")
def snapshot():
    async def snap():
        c = await get_cam()
        return await c.get_snapshot(channel=CHANNEL)

    jpg = run_async(snap())
    return Response(jpg, mimetype="image/jpeg")

@app.post("/ptz")
def ptz():
    data = request.get_json(force=True)
    op = data.get("op", "Stop")
    speed = int(data.get("speed", 20))

    async def do_ptz():
        c = await get_cam()
        await c.ptz_control(command=op, speed=speed, channel=CHANNEL)

    run_async(do_ptz())
    return jsonify({"ok": True, "op": op})

@app.post("/detect_dog")
def detect_dog():
    async def snap():
        c = await get_cam()
        return await c.get_snapshot(channel=CHANNEL)

    jpg = run_async(snap())
    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

    results = yolo.predict(frame, imgsz=640, conf=0.35, verbose=False)[0]
    dog_confs = [float(b.conf) for b in results.boxes if int(b.cls) == 16]  # COCO dog=16

    found = len(dog_confs) > 0
    conf = max(dog_confs) if found else 0.0
    return jsonify({"found": found, "confidence": conf})

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Reolink Dog Finder</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 20px; display: grid; gap: 12px; }
    #view { width: 800px; max-width: 100%; border-radius: 12px; border: 1px solid #ccc; }
    .controls { display: grid; grid-template-columns: repeat(3, 80px); gap: 8px; width: max-content; }
    button { padding: 10px; border-radius: 10px; border: 1px solid #aaa; background: #f7f7f7; cursor: pointer; }
    button:hover { background: #eee; }
    .row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
    #status { font-weight: 600; }
  </style>
</head>
<body>
  <h2>Reolink Dog Finder</h2>

  <img id="view" src="/snapshot.jpg" />

  <div class="row">
    <div class="controls">
      <div></div>
      <button onclick="ptz('Up')">▲</button>
      <div></div>

      <button onclick="ptz('Left')">◀</button>
      <button onclick="ptz('Stop')">■</button>
      <button onclick="ptz('Right')">▶</button>

      <div></div>
      <button onclick="ptz('Down')">▼</button>
      <div></div>
    </div>

    <div class="row">
      <button onclick="ptz('ZoomIn')">Zoom +</button>
      <button onclick="ptz('ZoomOut')">Zoom −</button>
      <button onclick="detectDog()">Scan for dog</button>
    </div>
  </div>

  <div id="status">Status: idle</div>

<script>
  // Refresh snapshot 2x/sec for "live-ish" view
  const img = document.getElementById("view");
  setInterval(() => {
    img.src = "/snapshot.jpg?t=" + Date.now();
  }, 500);

  async function ptz(op) {
    await fetch("/ptz", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({op, speed: 20})
    });
  }

  async function detectDog() {
    const status = document.getElementById("status");
    status.textContent = "Status: scanning…";

    const res = await fetch("/detect_dog", {method: "POST"});
    const data = await res.json();
    if (data.found) {
      status.textContent = `Status: DOG FOUND (conf ${data.confidence.toFixed(2)})`;
    } else {
      status.textContent = "Status: no dog in frame";
    }
  }
</script>
</body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(INDEX_HTML)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
