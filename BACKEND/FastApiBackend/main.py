import cv2
import os
import uuid
import time
import tkinter as tk
from tkinter import simpledialog

# === Setup ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)

# Constants
IMAGES_PER_PERSON = 5
CAPTURE_DELAY = 3  # seconds

# Tkinter setup (for getting the name later)
root = tk.Tk()
root.withdraw()

# Initialize webcam
cap = cv2.VideoCapture(0)
seen_hashes = set()
captured_faces = []
capture_count = 0

print("📸 Face capture running. Please position a new person in front of the camera.")

while capture_count < IMAGES_PER_PERSON:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        face_hash = hash(roi.tobytes())

        # Skip if we've already seen this face (avoids duplicates in same frame)
        if face_hash in seen_hashes:
            continue
        seen_hashes.add(face_hash)

        # Draw rectangle for visibility
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Capture", frame)
        cv2.waitKey(1)

        print(f"📷 Capturing image {capture_count + 1}/{IMAGES_PER_PERSON}...")
        time.sleep(CAPTURE_DELAY)

        # Re-capture after delay to get steady image
        ret2, frame_delay = cap.read()
        gray_delay = cv2.cvtColor(frame_delay, cv2.COLOR_BGR2GRAY)
        faces_delay = face_cascade.detectMultiScale(gray_delay, 1.1, 5)

        if len(faces_delay) > 0:
            (x2, y2, w2, h2) = faces_delay[0]
            roi_final = gray_delay[y2:y2 + h2, x2:x2 + w2]
            captured_faces.append(roi_final)
            capture_count += 1
        else:
            print("⚠️ Face not detected after delay.")

        break  # Process only one new face per frame

    cv2.imshow("Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("❎ Quit requested.")
        break

# Release camera
cap.release()
cv2.destroyAllWindows()

# Ask for name and save images
if captured_faces:
    name = simpledialog.askstring("Face Captured", "Enter name for this person:")
    if name:
        person_dir = os.path.join(dataset_dir, name)
        os.makedirs(person_dir, exist_ok=True)

        for idx, face_img in enumerate(captured_faces):
            filename = os.path.join(person_dir, f"{uuid.uuid4().hex[:8]}.jpg")
            cv2.imwrite(filename, face_img)
            print(f"✅ Saved {filename}")

        print(f"✅ Done: Saved {IMAGES_PER_PERSON} images for {name}")
    else:
        print("⚠️ No name entered. Images not saved.")
else:
    print("⚠️ No faces were captured.")

# Model
import cv2
import os
import numpy as np
import pickle

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load dataset
dataset_path = "dataset"
faces = []
labels = []
label_map = {}
label_id = 0

for name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, name)
    if not os.path.isdir(person_dir):
        continue
    label_map[label_id] = name

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        detected = face_cascade.detectMultiScale(img, 1.1, 5)
        for (x, y, w, h) in detected:
            faces.append(img[y:y + h, x:x + w])
            labels.append(label_id)
            break
    label_id += 1

if not faces:
    print("❌ No faces found.")
    exit()

# Train model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save("trained_model.yml")
with open("labels.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("✅ Model trained. Starting recognition...")

# Start recognition
cap = cv2.VideoCapture(0)
recognizer.read("trained_model.yml")
with open("labels.pkl", "rb") as f:
    label_map = pickle.load(f)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces_detected:
        roi = gray[y:y + h, x:x + w]
        label, confidence = recognizer.predict(roi)

        name = "Unknown"
        if confidence < 70:
            name = label_map.get(label, "Unknown")

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{name}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# # from fastapi import FastAPI, Request
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.responses import HTMLResponse
# # from fastapi.templating import Jinja2Templates
# # import socketio
# # import uvicorn
# # import os
# # from pathlib import Path
# # import logging

# # # Setup logging
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger("video-stream")

# # # Create FastAPI app
# # app = FastAPI(title="Simple Video Stream")

# # # Enable CORS
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # # Create Socket.IO server
# # sio = socketio.AsyncServer(
# #     async_mode='asgi',
# #     cors_allowed_origins='*',
# #     logger=True
# # )

# # # Create ASGI app
# # socket_app = socketio.ASGIApp(sio, app)

# # # Create templates directory
# # templates_dir = Path("templates")
# # templates_dir.mkdir(exist_ok=True)
# # templates = Jinja2Templates(directory="templates")

# # # Create viewer.html template
# # with open("templates/viewer.html", "w") as f:
# #     f.write("""
# # <!DOCTYPE html>
# # <html>
# # <head>
# #     <title>Live Video Stream Viewer</title>
# #     <style>
# #         body {
# #             font-family: Arial, sans-serif;
# #             margin: 0;
# #             padding: 20px;
# #             background: #f0f0f0;
# #             text-align: center;
# #         }
# #         .container {
# #             max-width: 800px;
# #             margin: 0 auto;
# #             background: white;
# #             padding: 20px;
# #             border-radius: 10px;
# #             box-shadow: 0 2px 10px rgba(0,0,0,0.1);
# #         }
# #         h1 { margin-bottom: 20px; }
# #         #videoDisplay {
# #             width: 100%;
# #             max-width: 640px;
# #             height: auto;
# #             border: 1px solid #ddd;
# #             border-radius: 4px;
# #             background: #000;
# #         }
# #         .status {
# #             margin-top: 10px;
# #             padding: 5px;
# #             border-radius: 4px;
# #         }
# #         .connected { background: #d4edda; color: #155724; }
# #         .disconnected { background: #f8d7da; color: #721c24; }
# #         #fps, #frameInfo { font-size: 14px; margin-top: 10px; }
# #     </style>
# #     <script src="https://cdn.socket.io/4.6.0/socket.io.min.js"></script>
# # </head>
# # <body>
# #     <div class="container">
# #         <h1>Live Video Stream</h1>
# #         <img id="videoDisplay" src="data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==" alt="Waiting for video...">
# #         <div id="status" class="status disconnected">Waiting for connection...</div>
# #         <div id="fps">FPS: 0</div>
# #         <div id="frameInfo">Waiting for frames...</div>
# #     </div>

# #     <script>
# #         const videoDisplay = document.getElementById('videoDisplay');
# #         const statusElement = document.getElementById('status');
# #         const fpsElement = document.getElementById('fps');
# #         const frameInfoElement = document.getElementById('frameInfo');
        
# #         // Track frames for FPS calculation
# #         let frameCount = 0;
# #         let lastFrameTime = Date.now();
# #         let framesReceived = 0;
        
# #         // Connect to Socket.IO server
# #         const socket = io({
# #             reconnectionDelay: 1000,
# #             reconnectionDelayMax: 5000,
# #             reconnectionAttempts: Infinity
# #         });
        
# #         // Handle connection events
# #         socket.on('connect', () => {
# #             console.log('Connected to server with ID:', socket.id);
# #             statusElement.textContent = 'Connected. Waiting for video...';
# #             statusElement.className = 'status connected';
            
# #             // Register as viewer
# #             socket.emit('register_viewer');
# #         });
        
# #         socket.on('disconnect', () => {
# #             console.log('Disconnected from server');
# #             statusElement.textContent = 'Disconnected from server';
# #             statusElement.className = 'status disconnected';
# #         });
        
# #         // Debug connection issues
# #         socket.on('connect_error', (err) => {
# #             console.error('Connection error:', err);
# #             statusElement.textContent = 'Connection error: ' + err.message;
# #             statusElement.className = 'status disconnected';
# #         });
        
# #         // Handle incoming frames
# #         socket.on('frame', (data) => {
# #             // Check if we have frame data
# #             if (!data.frame) {
# #                 console.error('Received empty frame data');
# #                 return;
# #             }
            
# #             // Update frame counter
# #             framesReceived++;
# #             frameCount++;
# #             const now = Date.now();
            
# #             // Calculate FPS every second
# #             if (now - lastFrameTime >= 1000) {
# #                 const fps = Math.round(frameCount * 1000 / (now - lastFrameTime));
# #                 fpsElement.textContent = `FPS: ${fps}`;
# #                 frameCount = 0;
# #                 lastFrameTime = now;
# #             }
            
# #             // Update image with received frame
# #             try {
# #                 videoDisplay.src = 'data:image/jpeg;base64,' + data.frame;
# #             } catch (e) {
# #                 console.error('Error setting image source:', e);
# #             }
            
# #             // Update frame info
# #             frameInfoElement.textContent = `Frames received: ${framesReceived}, Last update: ${new Date().toLocaleTimeString()}`;
            
# #             // Update status
# #             statusElement.textContent = 'Connected - Receiving Video Stream';
# #             statusElement.className = 'status connected';
            
# #             // Send acknowledgment back to server
# #             socket.emit('frame_received', { timestamp: now });
# #         });
# #     </script>
# # </body>
# # </html>
# #     """)

# # # Simple in-memory storage
# # class StreamManager:
# #     def __init__(self):
# #         self.broadcasters = set()
# #         self.viewers = set()
# #         self.last_frame = None
    
# #     def add_broadcaster(self, sid):
# #         self.broadcasters.add(sid)
# #         logger.info(f"Broadcaster connected: {sid}")
    
# #     def remove_broadcaster(self, sid):
# #         self.broadcasters.discard(sid)
# #         logger.info(f"Broadcaster disconnected: {sid}")
    
# #     def add_viewer(self, sid):
# #         self.viewers.add(sid)
# #         logger.info(f"Viewer connected: {sid}")
    
# #     def remove_viewer(self, sid):
# #         self.viewers.discard(sid)
# #         logger.info(f"Viewer disconnected: {sid}")

# # # Create manager
# # manager = StreamManager()

# # # Socket.IO event handlers
# # @sio.event
# # async def connect(sid, environ):
# #     logger.info(f"Client connected: {sid}")

# # @sio.event
# # async def disconnect(sid):
# #     if sid in manager.broadcasters:
# #         manager.remove_broadcaster(sid)
# #     if sid in manager.viewers:
# #         manager.remove_viewer(sid)

# # @sio.event
# # async def register_broadcaster(sid, data=None):
# #     manager.add_broadcaster(sid)
# #     return {"status": "success"}

# # @sio.event
# # async def register_viewer(sid):
# #     manager.add_viewer(sid)
    
# #     # Send the last frame if available
# #     if manager.last_frame:
# #         await sio.emit('frame', {
# #             "frame": manager.last_frame,
# #             "timestamp": "from_cache"
# #         }, room=sid)
    
# #     return {"status": "success"}

# # @sio.event
# # async def frame(sid, data):
# #     if sid not in manager.broadcasters:
# #         return {"status": "error", "message": "Not registered as broadcaster"}
    
# #     # Get the frame data
# #     frame_data = data.get("frame")
# #     if not frame_data:
# #         return {"status": "error", "message": "No frame data"}
    
# #     # Store the last frame
# #     manager.last_frame = frame_data
    
# #     # Send to all viewers - FIXED: Use room parameter properly
# #     if manager.viewers:
# #         # Debug log
# #         logger.info(f"Broadcasting frame to {len(manager.viewers)} viewers")
        
# #         # Loop through each viewer and emit individually for reliability
# #         for viewer_sid in manager.viewers:
# #             try:
# #                 await sio.emit('frame', {
# #                     "frame": frame_data,
# #                     "timestamp": data.get("timestamp", "")
# #                 }, room=viewer_sid)
# #             except Exception as e:
# #                 logger.error(f"Error sending frame to {viewer_sid}: {str(e)}")
    
# #     return {"status": "success", "viewers": len(manager.viewers)}

# # # FastAPI routes
# # @app.get("/", response_class=HTMLResponse)
# # async def get_index(request: Request):
# #     return templates.TemplateResponse("viewer.html", {"request": request})

# # @app.get("/status")
# # async def get_status():
# #     return {
# #         "broadcasters": len(manager.broadcasters),
# #         "viewers": len(manager.viewers),
# #         "has_frame": manager.last_frame is not None
# #     }

# # # Run the app
# # if __name__ == "__main__":
# #     uvicorn.run(socket_app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# import socketio
# import uvicorn
# import os
# from pathlib import Path
# import logging
# import cv2
# import numpy as np
# import base64
# from datetime import datetime
# import time
# import uuid
# import json
# import urllib.request

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("video-stream")

# # Create FastAPI app
# app = FastAPI(title="Face Detection Video Stream")

# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Create Socket.IO server
# sio = socketio.AsyncServer(
#     async_mode='asgi',
#     cors_allowed_origins='*',
#     logger=True
# )

# # Create ASGI app
# socket_app = socketio.ASGIApp(sio, app)

# # Create necessary directories
# templates_dir = Path("templates")
# templates_dir.mkdir(exist_ok=True)
# templates = Jinja2Templates(directory="templates")

# dataset_dir = Path("dataset")
# dataset_dir.mkdir(exist_ok=True)


# # Face Detection Helper
# class FaceDetector:
#     def __init__(self):
#         self.face_cascade = None
#         self.detection_cooldown = 0.5  # seconds between detection attempts
#         self.last_detection = 0
#         self.load_cascade()
        
#     def load_cascade(self):
#         """Load face detection cascade with multiple fallbacks"""
#         # Try OpenCV's default location
#         try:
#             cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
#             self.face_cascade = cv2.CascadeClassifier(cascade_path)
#             if not self.face_cascade.empty():
#                 logger.info("Loaded face cascade from OpenCV")
#                 return True
#         except Exception as e:
#             logger.warning(f"Failed to load cascade from OpenCV: {e}")
        
#         # Check for local file
#         local_cascade = "haarcascade_frontalface_default.xml"
#         if os.path.exists(local_cascade):
#             self.face_cascade = cv2.CascadeClassifier(local_cascade)
#             if not self.face_cascade.empty():
#                 logger.info("Loaded face cascade from local file")
#                 return True
        
#         # Download if not found
#         try:
#             logger.info("Downloading face cascade file...")
#             url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
#             urllib.request.urlretrieve(url, local_cascade)
            
#             self.face_cascade = cv2.CascadeClassifier(local_cascade)
#             if not self.face_cascade.empty():
#                 logger.info("Successfully downloaded face cascade file")
#                 return True
#         except Exception as e:
#             logger.error(f"Failed to download cascade: {e}")
        
#         logger.error("Could not load face cascade. Face detection will not work.")
#         return False
    
#     def detect_faces(self, frame_data):
#         """Detect faces in a base64 encoded frame"""
#         # Skip if face detection isn't available
#         if not self.face_cascade or self.face_cascade.empty():
#             return None, None
        
#         # Rate limit detection for performance
#         current_time = time.time()
#         if current_time - self.last_detection < self.detection_cooldown:
#             return None, None
            
#         self.last_detection = current_time
        
#         try:
#             # Decode base64 image
#             if "base64," in frame_data:
#                 frame_data = frame_data.split("base64,")[1]
                
#             image_bytes = base64.b64decode(frame_data)
#             nparr = np.frombuffer(image_bytes, np.uint8)
#             frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
#             if frame is None:
#                 logger.error("Failed to decode image")
#                 return None, None
                
#             # Convert to grayscale for face detection
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
#             # Detect faces
#             faces = self.face_cascade.detectMultiScale(
#                 gray, 
#                 scaleFactor=1.1,
#                 minNeighbors=5,
#                 minSize=(30, 30)
#             )
            
#             # No faces detected
#             if len(faces) == 0:
#                 return [], None
                
#             # Create a copy for drawing
#             vis_frame = frame.copy()
            
#             # Process detected faces
#             results = []
#             for (x, y, w, h) in faces:
#                 # Create result object
#                 face_id = str(uuid.uuid4())[:8]
#                 result = {
#                     "id": face_id,
#                     "box": [int(x), int(y), int(w), int(h)],
#                     "confidence": 95.0,
#                     "name": "Person",
#                     "timestamp": datetime.now().isoformat()
#                 }
#                 results.append(result)
                
#                 # Draw rectangle and label
#                 cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                 cv2.putText(vis_frame, "Person", (x, y-10), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
#             # Encode back to base64
#             _, buffer = cv2.imencode('.jpg', vis_frame)
#             processed_frame = base64.b64encode(buffer).decode('utf-8')
            
#             return results, processed_frame
            
#         except Exception as e:
#             logger.error(f"Error in face detection: {e}")
#             return None, None


# # Person database for storing recognized people
# class PersonDatabase:
#     def __init__(self):
#         self.people = {}
#         self.load_database()
        
#     def load_database(self):
#         """Load saved people from disk"""
#         db_file = "people.json"
#         if os.path.exists(db_file):
#             try:
#                 with open(db_file, 'r') as f:
#                     self.people = json.load(f)
#                 logger.info(f"Loaded {len(self.people)} people from database")
#             except Exception as e:
#                 logger.error(f"Failed to load people database: {e}")
#                 self.people = {}
#         else:
#             self.people = {}
            
#     def save_database(self):
#         """Save people database to disk"""
#         try:
#             with open("people.json", 'w') as f:
#                 json.dump(self.people, f)
#             logger.info(f"Saved {len(self.people)} people to database")
#             return True
#         except Exception as e:
#             logger.error(f"Failed to save people database: {e}")
#             return False
            
#     def add_person(self, person_id, name, relationship="", description=""):
#         """Add a new person to the database"""
#         if not person_id or not name:
#             return False
            
#         self.people[person_id] = {
#             "id": person_id,
#             "name": name,
#             "relationship": relationship,
#             "description": description,
#             "added": datetime.now().isoformat(),
#             "lastSeen": datetime.now().isoformat(),
#             "imageUrl": f"/people/{person_id}.jpg"
#         }
        
#         self.save_database()
#         return True
        
#     def update_last_seen(self, person_id):
#         """Update when a person was last seen"""
#         if person_id in self.people:
#             self.people[person_id]["lastSeen"] = datetime.now().isoformat()
#             return True
#         return False
        
#     def get_person(self, person_id):
#         """Get person by ID"""
#         return self.people.get(person_id)
        
#     def get_all_people(self):
#         """Get all people in the database"""
#         return list(self.people.values())


# # Enhanced StreamManager with face detection
# class StreamManager:
#     def __init__(self):
#         self.broadcasters = set()
#         self.viewers = set()
#         self.last_frame = None
#         self.face_detector = FaceDetector()
#         self.person_db = PersonDatabase()
#         self.detection_enabled = True
#         self.pending_faces = {}  # Faces waiting for name assignment
#         self.recently_detected = {}  # To avoid duplicate notifications
        
#     def add_broadcaster(self, sid):
#         self.broadcasters.add(sid)
#         logger.info(f"Broadcaster connected: {sid}")
    
#     def remove_broadcaster(self, sid):
#         self.broadcasters.discard(sid)
#         logger.info(f"Broadcaster disconnected: {sid}")
    
#     def add_viewer(self, sid):
#         self.viewers.add(sid)
#         logger.info(f"Viewer connected: {sid}")
    
#     def remove_viewer(self, sid):
#         self.viewers.discard(sid)
#         logger.info(f"Viewer disconnected: {sid}")
        
#     async def process_frame(self, frame_data):
#         """Process a frame for face detection"""
#         if not self.detection_enabled:
#             return None, None
            
#         # Detect faces
#         face_results, processed_frame = self.face_detector.detect_faces(frame_data)
        
#         # If we detected faces, save them as pending
#         if face_results:
#             current_time = time.time()
            
#             for result in face_results:
#                 face_id = result["id"]
                
#                 # Store as a pending face if not recently seen
#                 if face_id not in self.recently_detected:
#                     self.pending_faces[face_id] = {
#                         "result": result,
#                         "detected_at": current_time,
#                         "frame": frame_data  # Store original frame
#                     }
                    
#                     # Remember this face to avoid spam
#                     self.recently_detected[face_id] = current_time
                    
#                     # Flag it as needing registration
#                     result["needs_registration"] = True
                
#                 # Clear old entries from recently_detected (after 60 seconds)
#                 self.recently_detected = {
#                     fid: t for fid, t in self.recently_detected.items() 
#                     if current_time - t < 60
#                 }
        
#         return face_results, processed_frame
    
#     def save_person(self, face_id, name, relationship="", description=""):
#         """Save a detected person with provided details"""
#         if face_id not in self.pending_faces:
#             logger.error(f"Face ID {face_id} not found in pending faces")
#             return False
            
#         # Add to database
#         success = self.person_db.add_person(face_id, name, relationship, description)
        
#         # Save the face image if needed
#         # You could add code here to extract and save the face image
        
#         # Clean up
#         if success:
#             self.pending_faces.pop(face_id, None)
            
#         return success


# # Initialize manager
# manager = StreamManager()


# # Socket.IO event handlers
# @sio.event
# async def connect(sid, environ):
#     logger.info(f"Client connected: {sid}")

# @sio.event
# async def disconnect(sid):
#     if sid in manager.broadcasters:
#         manager.remove_broadcaster(sid)
#     if sid in manager.viewers:
#         manager.remove_viewer(sid)

# @sio.event
# async def register_broadcaster(sid, data=None):
#     manager.add_broadcaster(sid)
#     return {"status": "success"}

# @sio.event
# async def register_viewer(sid):
#     manager.add_viewer(sid)
    
#     # Send the last frame if available
#     if manager.last_frame:
#         await sio.emit('frame', {
#             "frame": manager.last_frame,
#             "timestamp": "from_cache"
#         }, room=sid)
    
#     return {"status": "success"}

# @sio.event
# async def frame(sid, data):
#     if sid not in manager.broadcasters:
#         return {"status": "error", "message": "Not registered as broadcaster"}
    
#     # Get the frame data
#     frame_data = data.get("frame")
#     if not frame_data:
#         return {"status": "error", "message": "No frame data"}
    
#     # Store the last frame
#     manager.last_frame = frame_data
    
#     # Process the frame if requested
#     processed_frame = None
#     if data.get("process", False):
#         face_results, processed_frame = await manager.process_frame(frame_data)
        
#         # If we detected faces, notify the broadcaster
#         if face_results:
#             await sio.emit('detection_results', {
#                 "faces": face_results,
#                 "timestamp": datetime.now().isoformat()
#             }, room=sid)
            
#             # Check for faces needing registration
#             for result in face_results:
#                 if result.get("needs_registration", False):
#                     await sio.emit('new_person_detected', {
#                         "face_id": result["id"],
#                         "timestamp": datetime.now().isoformat()
#                     }, room=sid)
    
#     # Send to all viewers
#     if manager.viewers:
#         # Use processed frame if available
#         frame_to_send = processed_frame if processed_frame else frame_data
        
#         logger.info(f"Broadcasting frame to {len(manager.viewers)} viewers")
        
#         # Loop through each viewer and emit individually
#         for viewer_sid in manager.viewers:
#             try:
#                 await sio.emit('frame', {
#                     "frame": frame_to_send,
#                     "timestamp": data.get("timestamp", "")
#                 }, room=viewer_sid)
#             except Exception as e:
#                 logger.error(f"Error sending frame to {viewer_sid}: {str(e)}")
    
#     return {"status": "success", "viewers": len(manager.viewers)}

# @sio.event
# async def save_person(sid, data):
#     """Save a detected person"""
#     face_id = data.get("face_id")
#     name = data.get("name")
#     relationship = data.get("relationship", "")
#     description = data.get("description", "")
    
#     if not face_id or not name:
#         return {"status": "error", "message": "Missing required data"}
        
#     success = manager.save_person(face_id, name, relationship, description)
    
#     if success:
#         # Notify the broadcaster that a person was saved
#         await sio.emit('person_saved', {
#             "id": face_id,
#             "name": name,
#             "timestamp": datetime.now().isoformat()
#         }, room=sid)
        
#     return {"status": "success" if success else "error"}

# @sio.event
# async def toggle_detection(sid, data):
#     """Toggle face detection on/off"""
#     enabled = data.get("enabled", True)
#     manager.detection_enabled = enabled
#     logger.info(f"Face detection {'enabled' if enabled else 'disabled'}")
#     return {"status": "success", "detection_enabled": manager.detection_enabled}

# @sio.event
# async def get_people(sid):
#     """Get all saved people"""
#     return {
#         "status": "success",
#         "people": manager.person_db.get_all_people()
#     }

# @sio.event 
# async def ping(sid):
#     """Simple ping to test connectivity"""
#     return {"status": "success", "time": datetime.now().isoformat()}


# # Create viewer.html template
# with open("templates/viewer.html", "w") as f:
#     f.write("""
# <!DOCTYPE html>
# <html>
# <head>
#     <title>Face Detection Video Stream</title>
#     <style>
#         body {
#             font-family: Arial, sans-serif;
#             margin: 0;
#             padding: 20px;
#             background: #f0f0f0;
#             text-align: center;
#         }
#         .container {
#             max-width: 800px;
#             margin: 0 auto;
#             background: white;
#             padding: 20px;
#             border-radius: 10px;
#             box-shadow: 0 2px 10px rgba(0,0,0,0.1);
#         }
#         h1 { margin-bottom: 20px; }
#         #videoDisplay {
#             width: 100%;
#             max-width: 640px;
#             height: auto;
#             border: 1px solid #ddd;
#             border-radius: 4px;
#             background: #000;
#         }
#         .status {
#             margin-top: 10px;
#             padding: 5px;
#             border-radius: 4px;
#         }
#         .connected { background: #d4edda; color: #155724; }
#         .disconnected { background: #f8d7da; color: #721c24; }
#         #fps, #frameInfo { font-size: 14px; margin-top: 10px; }
#     </style>
#     <script src="https://cdn.socket.io/4.6.0/socket.io.min.js"></script>
# </head>
# <body>
#     <div class="container">
#         <h1>Face Detection Video Stream</h1>
#         <img id="videoDisplay" src="data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==" alt="Waiting for video...">
#         <div id="status" class="status disconnected">Waiting for connection...</div>
#         <div id="fps">FPS: 0</div>
#         <div id="frameInfo">Waiting for frames...</div>
#         <div id="faceInfo"></div>
#     </div>

#     <script>
#         const videoDisplay = document.getElementById('videoDisplay');
#         const statusElement = document.getElementById('status');
#         const fpsElement = document.getElementById('fps');
#         const frameInfoElement = document.getElementById('frameInfo');
#         const faceInfoElement = document.getElementById('faceInfo');
        
#         // Track frames for FPS calculation
#         let frameCount = 0;
#         let lastFrameTime = Date.now();
#         let framesReceived = 0;
        
#         // Connect to Socket.IO server
#         const socket = io({
#             reconnectionDelay: 1000,
#             reconnectionDelayMax: 5000,
#             reconnectionAttempts: Infinity
#         });
        
#         // Handle connection events
#         socket.on('connect', () => {
#             console.log('Connected to server with ID:', socket.id);
#             statusElement.textContent = 'Connected. Waiting for video...';
#             statusElement.className = 'status connected';
            
#             // Register as viewer
#             socket.emit('register_viewer');
#         });
        
#         socket.on('disconnect', () => {
#             console.log('Disconnected from server');
#             statusElement.textContent = 'Disconnected from server';
#             statusElement.className = 'status disconnected';
#         });
        
#         // Debug connection issues
#         socket.on('connect_error', (err) => {
#             console.error('Connection error:', err);
#             statusElement.textContent = 'Connection error: ' + err.message;
#             statusElement.className = 'status disconnected';
#         });
        
#         // Handle incoming frames
#         socket.on('frame', (data) => {
#             // Check if we have frame data
#             if (!data.frame) {
#                 console.error('Received empty frame data');
#                 return;
#             }
            
#             // Update frame counter
#             framesReceived++;
#             frameCount++;
#             const now = Date.now();
            
#             // Calculate FPS every second
#             if (now - lastFrameTime >= 1000) {
#                 const fps = Math.round(frameCount * 1000 / (now - lastFrameTime));
#                 fpsElement.textContent = `FPS: ${fps}`;
#                 frameCount = 0;
#                 lastFrameTime = now;
#             }
            
#             // Update image with received frame
#             try {
#                 videoDisplay.src = 'data:image/jpeg;base64,' + data.frame;
#             } catch (e) {
#                 console.error('Error setting image source:', e);
#             }
            
#             // Update frame info
#             frameInfoElement.textContent = `Frames received: ${framesReceived}, Last update: ${new Date().toLocaleTimeString()}`;
            
#             // Update status
#             statusElement.textContent = 'Connected - Receiving Video Stream';
#             statusElement.className = 'status connected';
            
#             // Send acknowledgment back to server
#             socket.emit('frame_received', { timestamp: now });
#         });
        
#         // Handle face detection info
#         socket.on('face_detected', (data) => {
#             faceInfoElement.textContent = `Face detected: ${new Date().toLocaleTimeString()}`;
#         });
#     </script>
# </body>
# </html>
#     """)


# # FastAPI routes
# @app.get("/", response_class=HTMLResponse)
# async def get_index(request: Request):
#     return templates.TemplateResponse("viewer.html", {"request": request})

# @app.get("/status")
# async def get_status():
#     return {
#         "status": "running",
#         "broadcasters": len(manager.broadcasters),
#         "viewers": len(manager.viewers),
#         "has_frame": manager.last_frame is not None,
#         "face_detection": {
#             "available": manager.face_detector.face_cascade is not None and not manager.face_detector.face_cascade.empty(),
#             "enabled": manager.detection_enabled,
#             "pending_faces": len(manager.pending_faces)
#         },
#         "database": {
#             "people_count": len(manager.person_db.get_all_people())
#         },
#         "timestamp": datetime.now().isoformat()
#     }

# # Run the app
# if __name__ == "__main__":
#     uvicorn.run(socket_app, host="0.0.0.0", port=8000)

