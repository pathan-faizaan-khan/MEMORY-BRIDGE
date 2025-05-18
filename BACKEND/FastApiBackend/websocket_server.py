from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import socketio
import uvicorn
import os
import numpy as np
import cv2
import base64
import uuid
import pickle
import json
from pathlib import Path
import logging
import time
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("video-stream")

# Create FastAPI app
app = FastAPI(title="Face Recognition Stream")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create Socket.IO server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    logger=True
)

# Create ASGI app
socket_app = socketio.ASGIApp(sio, app)

# Create directories
templates_dir = Path("templates")
templates_dir.mkdir(exist_ok=True)
templates = Jinja2Templates(directory="templates")

dataset_dir = Path("dataset")
dataset_dir.mkdir(exist_ok=True)

# Face Recognition Manager
class FaceRecognitionManager:
    def __init__(self):
        # Load face cascade classifier
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                raise ValueError("Failed to load cascade classifier")
        except Exception as e:
            logger.error(f"Error loading face cascade: {e}")
            # Try downloading the cascade file
            import urllib.request
            url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
            urllib.request.urlretrieve(url, "haarcascade_frontalface_default.xml")
            self.face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        
        # Initialize face recognizer
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.has_recognizer = True
        except:
            logger.warning("OpenCV face recognition module not available. Will only do detection.")
            self.has_recognizer = False
        
        # Face collection variables
        self.collecting_faces = False
        self.current_person_faces = []
        self.min_faces_needed = 5
        self.face_collection_cooldown = 0.5  # seconds between captures
        self.last_capture_time = 0
        
        # Recognition variables
        self.label_map = {}  # Maps label IDs to names
        self.confidence_threshold = 70  # Lower = stricter matching
        self.recognition_cooldown = 1.0  # seconds between recognition attempts
        self.last_recognition_time = 0
        
        # Recognition tracking
        self.recently_recognized = {}  # { name: last_recognized_time }
        
        # Load trained model if available
        self.load_recognizer()
        
    def load_recognizer(self):
        """Load the trained face recognition model if available"""
        if not self.has_recognizer:
            return False
            
        model_file = "face_model.yml"
        label_file = "face_labels.pkl"
        
        if os.path.exists(model_file) and os.path.exists(label_file):
            try:
                self.recognizer.read(model_file)
                with open(label_file, 'rb') as f:
                    self.label_map = pickle.load(f)
                logger.info(f"Loaded face model with {len(self.label_map)} people")
                return True
            except Exception as e:
                logger.error(f"Error loading face model: {e}")
        
        logger.info("No face model found. Will train when faces are collected.")
        return False
        
    def train_recognizer(self):
        """Train the face recognizer with collected face images"""
        if not self.has_recognizer:
            return False
            
        faces = []
        labels = []
        label_id = 0
        self.label_map = {}
        
        # Load all faces from the dataset directory
        for name in os.listdir(dataset_dir):
            person_dir = dataset_dir / name
            if not person_dir.is_dir():
                continue
                
            self.label_map[label_id] = name
            
            for img_file in os.listdir(person_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                img_path = person_dir / img_file
                face_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                
                if face_img is None:
                    continue
                    
                faces.append(face_img)
                labels.append(label_id)
                
            label_id += 1
        
        # Check if we have enough data to train
        if len(faces) < 5 or label_id == 0:
            logger.warning("Not enough face data to train recognizer")
            return False
        
        # Train the model
        logger.info(f"Training model with {len(faces)} faces of {label_id} people")
        self.recognizer.train(faces, np.array(labels))
        
        # Save the model
        self.recognizer.save("face_model.yml")
        with open("face_labels.pkl", 'wb') as f:
            pickle.dump(self.label_map, f)
            
        logger.info("Face recognition model trained and saved")
        return True
    
    def start_face_collection(self):
        """Start collecting faces for a new person"""
        self.collecting_faces = True
        self.current_person_faces = []
        logger.info("Started face collection")
        return True
        
    def cancel_face_collection(self):
        """Cancel the current face collection"""
        self.collecting_faces = False
        self.current_person_faces = []
        logger.info("Face collection cancelled")
        return True
        
    def save_collected_faces(self, name, relationship="", description=""):
        """Save collected faces with person details"""
        if not self.collecting_faces or len(self.current_person_faces) < self.min_faces_needed:
            return False
            
        if not name or name.strip() == "":
            return False
            
        # Create directory for this person
        person_dir = dataset_dir / name
        person_dir.mkdir(exist_ok=True)
        
        # Save each face image
        for idx, face_img in enumerate(self.current_person_faces):
            filename = f"{uuid.uuid4().hex[:8]}.jpg"
            filepath = person_dir / filename
            cv2.imwrite(str(filepath), face_img)
            
        # Save metadata
        metadata = {
            "name": name,
            "relationship": relationship,
            "description": description,
            "added_on": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat()
        }
        
        with open(person_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)
            
        # Reset collection state
        self.collecting_faces = False
        self.current_person_faces = []
        
        # Retrain the recognizer
        self.train_recognizer()
        
        logger.info(f"Saved {len(self.current_person_faces)} face images for {name}")
        return True
    
    def get_person_metadata(self, name):
        """Get metadata for a person"""
        if name == "Unknown":
            return {
                "name": "Unknown Person",
                "relationship": "",
                "description": "Person not recognized",
                "last_seen": datetime.now().isoformat()
            }
            
        person_dir = dataset_dir / name
        metadata_file = person_dir / "metadata.json"
        
        if not metadata_file.exists():
            return {
                "name": name,
                "relationship": "",
                "description": "",
                "last_seen": datetime.now().isoformat()
            }
            
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                
            # Update last seen
            metadata["last_seen"] = datetime.now().isoformat()
            
            with open(metadata_file, "w") as f:
                json.dump(metadata, f)
                
            return metadata
        except Exception as e:
            logger.error(f"Error reading metadata for {name}: {e}")
            return {
                "name": name,
                "relationship": "",
                "description": "",
                "last_seen": datetime.now().isoformat()
            }
    
    def process_frame(self, frame_data):
        """Process a frame for face detection and recognition"""
        # Skip if no cascade classifier
        if self.face_cascade.empty():
            logger.error("Face cascade classifier is empty")
            return None, None
        
        try:
            # Decode the base64 image
            if "base64," in frame_data:
                frame_data = frame_data.split("base64,")[1]
                
            image_bytes = base64.b64decode(frame_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.error("Failed to decode image data")
                return None, None
                
            # Log frame dimensions for debugging
            logger.info(f"Processing frame: {frame.shape}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Log face detection results
            if len(faces) == 0:
                logger.info("No faces detected in frame")
                return [], None
            else:
                logger.info(f"Detected {len(faces)} faces in frame")
            
            # No faces found
            if len(faces) == 0:
                return [], None
                
            # Create a copy for drawing
            vis_frame = frame.copy()
            
            # Face results
            results = []
            current_time = time.time()
            
            # Process each face
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                
                # Store for face collection if active
                if self.collecting_faces and current_time - self.last_capture_time >= self.face_collection_cooldown:
                    self.last_capture_time = current_time
                    if len(self.current_person_faces) < self.min_faces_needed:
                        # Resize face to a standard size for consistency
                        face_roi_resized = cv2.resize(face_roi, (100, 100))
                        self.current_person_faces.append(face_roi_resized)
                
                # Default result with unknown identity
                result = {
                    "id": str(uuid.uuid4())[:8],
                    "box": [int(x), int(y), int(w), int(h)],
                    "name": "Unknown",
                    "confidence": 0.0,
                    "recognized": False,
                    "is_new_detection": False
                }
                
                # Do face recognition if possible and not on cooldown
                if self.has_recognizer and len(self.label_map) > 0 and current_time - self.last_recognition_time >= self.recognition_cooldown:
                    self.last_recognition_time = current_time
                    
                    try:
                        # Predict the label
                        label_id, confidence = self.recognizer.predict(face_roi)
                        
                        # Lower confidence value = better match in OpenCV
                        recognition_confidence = 100 - confidence
                        result["confidence"] = float(recognition_confidence)
                        
                        if confidence < self.confidence_threshold:
                            name = self.label_map.get(label_id, "Unknown")
                            result["name"] = name
                            result["recognized"] = True
                            
                            # Get metadata
                            metadata = self.get_person_metadata(name)
                            result.update(metadata)
                            
                            # Check if this is a new detection (not seen recently)
                            if name not in self.recently_recognized or current_time - self.recently_recognized[name] > 10.0:
                                result["is_new_detection"] = True
                                self.recently_recognized[name] = current_time
                    except Exception as e:
                        logger.error(f"Error during face recognition: {e}")
                
                results.append(result)
                
                # Draw rectangle and label
                color = (0, 255, 0) if result["recognized"] else (0, 0, 255)
                cv2.rectangle(vis_frame, (x, y), (x+w, y+h), color, 2)
                
                label = result["name"]
                if result["recognized"]:
                    label += f" ({result['confidence']:.1f}%)";
                    
                cv2.putText(vis_frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Convert processed frame to base64
            _, buffer = cv2.imencode(".jpg", vis_frame)
            processed_frame = base64.b64encode(buffer).decode("utf-8")
            
            # Clean up old recognized entries (after 60 seconds)
            self.recently_recognized = {
                name: t for name, t in self.recently_recognized.items() 
                if current_time - t < 60
            }
            
            return results, processed_frame
        
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return None, None
            
    def get_collection_status(self):
        """Get the status of face collection"""
        return {
            "collecting": self.collecting_faces,
            "faces_collected": len(self.current_person_faces),
            "faces_needed": self.min_faces_needed,
            "progress": min(len(self.current_person_faces) / self.min_faces_needed * 100, 100)
        }

# Stream manager
class StreamManager:
    def __init__(self):
        self.broadcasters = set()
        self.viewers = set()
        self.last_frame = None
        self.face_manager = FaceRecognitionManager()
    
    def add_broadcaster(self, sid):
        self.broadcasters.add(sid)
        logger.info(f"Broadcaster connected: {sid}")
    
    def remove_broadcaster(self, sid):
        self.broadcasters.discard(sid)
        logger.info(f"Broadcaster disconnected: {sid}")
    
    def add_viewer(self, sid):
        self.viewers.add(sid)
        logger.info(f"Viewer connected: {sid}")
    
    def remove_viewer(self, sid):
        self.viewers.discard(sid)
        logger.info(f"Viewer disconnected: {sid}")

# Create manager
manager = StreamManager()

# Socket.IO event handlers
@sio.event
async def connect(sid, environ):
    logger.info(f"Client connected: {sid}")

@sio.event
async def disconnect(sid):
    if sid in manager.broadcasters:
        manager.remove_broadcaster(sid)
    if sid in manager.viewers:
        manager.remove_viewer(sid)

@sio.event
async def register_broadcaster(sid, data=None):
    manager.add_broadcaster(sid)
    return {"status": "success"}

@sio.event
async def register_viewer(sid):
    manager.add_viewer(sid)
    
    # Send the last frame if available
    if manager.last_frame:
        await sio.emit('frame', {
            "frame": manager.last_frame,
            "timestamp": "from_cache"
        }, room=sid)
    
    return {"status": "success"}

@sio.event
async def frame(sid, data):
    if sid not in manager.broadcasters:
        return {"status": "error", "message": "Not registered as broadcaster"}
    
    # Get the frame data
    frame_data = data.get("frame")
    if not frame_data:
        logger.warning("Received frame event without frame data")
        return {"status": "error", "message": "No frame data"}
    
    # Store the last frame
    manager.last_frame = frame_data
    
    # Process for face detection/recognition
    face_results, processed_frame = manager.face_manager.process_frame(frame_data)
    
    # Check if we're in face collection mode
    if manager.face_manager.collecting_faces:
        collection_status = manager.face_manager.get_collection_status()
        
        # If we've collected enough faces, notify client
        if collection_status["faces_collected"] >= collection_status["faces_needed"]:
            await sio.emit('enrollment_ready', {
                "status": "ready",
                "faces_collected": collection_status["faces_collected"]
            }, room=sid)
        else:
            # Update client on collection progress
            await sio.emit('enrollment_progress', collection_status, room=sid)
    
    # Send recognition results if available
    if face_results:
        await sio.emit('recognition_results', {
            "results": face_results,
            "timestamp": datetime.now().isoformat()
        }, room=sid)
        
        # Check for newly recognized people
        for result in face_results:
            if result.get("recognized", False) and result.get("is_new_detection", False):
                await sio.emit('person_recognized', {
                    "person": {
                        "id": result.get("id", ""),
                        "name": result["name"],
                        "confidence": result.get("confidence", 0)
                    }
                }, room=sid)
    
    # Use processed frame (with annotations) if available
    frame_to_send = processed_frame if processed_frame else frame_data
    
    # Send frame to all viewers
    if manager.viewers:
        for viewer_sid in manager.viewers:
            try:
                await sio.emit('frame', {
                    "frame": frame_to_send,
                    "timestamp": data.get("timestamp", "")
                }, room=viewer_sid)
            except Exception as e:
                logger.error(f"Error sending frame to {viewer_sid}: {str(e)}")
    
    return {
        "status": "success", 
        "viewers": len(manager.viewers),
        "collection_active": manager.face_manager.collecting_faces
    }

@sio.event
async def start_face_collection(sid):
    """Start collecting faces for a new person"""
    success = manager.face_manager.start_face_collection()
    return {"status": "success" if success else "error", "message": "Face collection started"}

@sio.event
async def cancel_face_collection(sid):
    """Cancel face collection"""
    success = manager.face_manager.cancel_face_collection()
    return {"status": "success" if success else "error", "message": "Face collection cancelled"}

@sio.event
async def save_new_person(sid, data):
    """Save a new person with the collected faces"""
    name = data.get("name")
    relationship = data.get("relationship", "")
    description = data.get("description", "")
    
    if not name:
        return {"status": "error", "message": "Name is required"}
        
    success = manager.face_manager.save_collected_faces(name, relationship, description)
    
    return {
        "status": "success" if success else "error", 
        "message": f"Person {name} saved" if success else "Failed to save person"
    }

@sio.event
async def get_collection_status(sid):
    """Get the current status of face collection"""
    status = manager.face_manager.get_collection_status()
    return {"status": "success", "collection": status}

@sio.event
async def train_recognizer(sid):
    """Force retrain the face recognizer"""
    success = manager.face_manager.train_recognizer()
    return {"status": "success" if success else "error", "message": "Training complete"}

@sio.event
async def start_face_enrollment(sid, data=None):
    """Start the face enrollment process for a new person"""
    logger.info("Starting face enrollment process")
    
    # Reset collection variables
    manager.face_manager.start_face_collection()
    
    # Tell the client we're ready to start collecting faces
    await sio.emit('enrollment_started', {
        "status": "collecting",
        "faces_needed": manager.face_manager.min_faces_needed
    }, room=sid)
    
    return {"status": "success", "message": "Face enrollment started"}

@sio.event
async def save_enrolled_face(sid, data):
    """Save the enrolled face data with a name"""
    name = data.get("name")
    relationship = data.get("relationship", "")
    description = data.get("description", "")
    
    if not name:
        return {"status": "error", "message": "Name is required"}
    
    # Save the faces we've collected
    success = manager.face_manager.save_collected_faces(name, relationship, description)
    
    if success:
        # Retrain model automatically after saving
        manager.face_manager.train_recognizer()
        await sio.emit('enrollment_complete', {
            "status": "success",
            "name": name
        }, room=sid)
        
    return {"status": "success" if success else "error", 
            "message": f"Saved faces for {name}" if success else "Failed to save faces"}

# Create viewer.html template
with open("templates/viewer.html", "w") as f:
    f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Face Recognition Stream</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f0f0f0;
            text-align: center;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { margin-bottom: 20px; }
        #videoDisplay {
            width: 100%;
            max-width: 640px;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: #000;
        }
        .status {
            margin-top: 10px;
            padding: 5px;
            border-radius: 4px;
        }
        .connected { background: #d4edda; color: #155724; }
        .disconnected { background: #f8d7da; color: #721c24; }
        #fps, #frameInfo, #faceInfo { font-size: 14px; margin-top: 10px; }
    </style>
    <script src="https://cdn.socket.io/4.6.0/socket.io.min.js"></script>
</head>
<body>
    <div class="container">
        <h1>Face Recognition Stream</h1>
        <img id="videoDisplay" src="data:image/gif;base64,R0lGODlhAQABAIAAAP///wAAACH5BAEAAAAALAAAAAABAAEAAAICRAEAOw==" alt="Waiting for video...">
        <div id="status" class="status disconnected">Waiting for connection...</div>
        <div id="fps">FPS: 0</div>
        <div id="frameInfo">Waiting for frames...</div>
        <div id="faceInfo"></div>
    </div>

    <script>
        const videoDisplay = document.getElementById('videoDisplay');
        const statusElement = document.getElementById('status');
        const fpsElement = document.getElementById('fps');
        const frameInfoElement = document.getElementById('frameInfo');
        const faceInfoElement = document.getElementById('faceInfo');
        
        // Track frames for FPS calculation
        let frameCount = 0;
        let lastFrameTime = Date.now();
        let framesReceived = 0;
        
        // Connect to Socket.IO server
        const socket = io({
            reconnectionDelay: 1000,
            reconnectionDelayMax: 5000,
            reconnectionAttempts: Infinity
        });
        
        // Handle connection events
        socket.on('connect', () => {
            console.log('Connected to server with ID:', socket.id);
            statusElement.textContent = 'Connected. Waiting for video...';
            statusElement.className = 'status connected';
            
            // Register as viewer
            socket.emit('register_viewer');
        });
        
        socket.on('disconnect', () => {
            console.log('Disconnected from server');
            statusElement.textContent = 'Disconnected from server';
            statusElement.className = 'status disconnected';
        });
        
        // Debug connection issues
        socket.on('connect_error', (err) => {
            console.error('Connection error:', err);
            statusElement.textContent = 'Connection error: ' + err.message;
            statusElement.className = 'status disconnected';
        });
        
        // Handle incoming frames
        socket.on('frame', (data) => {
            if (!data.frame) {
                console.error('Received empty frame data');
                return;
            }
            
            // Update frame counter
            framesReceived++;
            frameCount++;
            const now = Date.now();
            
            // Calculate FPS every second
            if (now - lastFrameTime >= 1000) {
                const fps = Math.round(frameCount * 1000 / (now - lastFrameTime));
                fpsElement.textContent = `FPS: ${fps}`;
                frameCount = 0;
                lastFrameTime = now;
            }
            
            // Update image
            videoDisplay.src = 'data:image/jpeg;base64,' + data.frame;
            
            // Update info
            frameInfoElement.textContent = `Frames received: ${framesReceived}`;
            statusElement.textContent = 'Connected - Receiving Video Stream';
            statusElement.className = 'status connected';
        });
        
        socket.on('face_detected', (data) => {
            faceInfoElement.textContent = `Face detected: ${data.name || 'Unknown'}`;
        });
        
        // Face enrollment system
        const startFaceEnrollment = () => {
            socket.emit('start_face_enrollment', {}, (response) => {
                if (response.status === 'success') {
                    showMessage('Please position your face in the camera');
                }
            });
        };

        // Listen for enrollment progress
        socket.on('enrollment_progress', (data) => {
            showMessage(`Collecting faces: ${data.faces_collected}/${data.faces_needed}`);
            updateProgressBar(data.progress);
        });

        // When enrollment is ready
        socket.on('enrollment_ready', () => {
            showMessage('All faces collected! Please enter your name:');
            showNamePrompt((name) => {
                socket.emit('save_enrolled_face', {
                    name: name,
                    relationship: 'Friend',  // Optional
                    description: 'Added via web interface'  // Optional
                });
            });
        });

        // When enrollment is complete
        socket.on('enrollment_complete', (data) => {
            showMessage(`Enrollment complete for ${data.name}!`);
        });

        // Listen for recognition results
        socket.on('recognition_results', (data) => {
            // Update UI with recognized faces
            updateRecognitionDisplay(data.results);
        });

        // When a person is recognized
        socket.on('person_recognized', (data) => {
            showNotification(`Recognized: ${data.person.name}`);
        });
    </script>
</body>
</html>
    """)
    
# FastAPI routes
@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse("viewer.html", {"request": request})

@app.get("/status")
async def get_status():
    return {
        "status": "running",
        "broadcasters": len(manager.broadcasters),
        "viewers": len(manager.viewers),
        "has_frame": manager.last_frame is not None,
        "face_recognition": {
            "enabled": manager.face_manager.has_recognizer,
            "people_count": len(manager.face_manager.label_map),
            "collecting_faces": manager.face_manager.collecting_faces,
            "faces_collected": len(manager.face_manager.current_person_faces)
        },
        "timestamp": datetime.now().isoformat()
    }

# Run the app
if __name__ == "__main__":
    # Ensure the dataset directory exists
    dataset_dir.mkdir(exist_ok=True)
    
    # Train face recognizer on startup if possible
    try:
        manager.face_manager.train_recognizer()
    except Exception as e:
        logger.error(f"Error training recognizer: {e}")
        
    # Run the server
    uvicorn.run(socket_app, host="0.0.0.0", port=8000)

