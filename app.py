from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os
import cv2
import numpy as np
from PIL import Image
import json
from datetime import datetime, timedelta
import base64
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import threading
import time
import torch
from flask_socketio import SocketIO, emit, join_room, leave_room
from collections import deque

# Try to import MediaPipe as primary face detection method
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe module loaded successfully")
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe module not available.")

# Try to import OpenCV contrib for face detection
try:
    # Check if opencv-contrib is available
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    OPENCV_FACE_AVAILABLE = True
    print("✅ OpenCV face detection available")
except:
    OPENCV_FACE_AVAILABLE = False
    print("⚠️  OpenCV face detection not available.")

# Legacy face_recognition support (optional)
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("✅ Legacy face_recognition module available")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠️  Legacy face_recognition module not available.")

# Add ONNX imports
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    print("✅ ONNX Runtime loaded successfully")
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  ONNX Runtime not available.")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///surveillance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['KNOWN_PEOPLE_FOLDER'] = 'known_people'
app.config['UNKNOWN_PEOPLE_FOLDER'] = 'unknown_people'

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['KNOWN_PEOPLE_FOLDER'], exist_ok=True)
os.makedirs(app.config['UNKNOWN_PEOPLE_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
CORS(app)

# Database Models
class KnownPerson(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    face_encoding = db.Column(db.Text, nullable=False)  # JSON string of face encoding
    image_path = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class UnknownPerson(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    face_encoding = db.Column(db.Text, nullable=False)  # JSON string of face encoding
    image_path = db.Column(db.String(200), nullable=False)
    detected_at = db.Column(db.DateTime, default=datetime.utcnow)
    alert_sent = db.Column(db.Boolean, default=False)

class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.String(500), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    person_id = db.Column(db.Integer, db.ForeignKey('unknown_person.id'))

# Initialize YOLO model
import torch
from ultralytics import YOLO

# Patch torch_safe_load to force weights_only=False
import ultralytics.nn.tasks as ul_tasks

def patched_safe_load(file):
    print("✅ Using patched torch.load(weights_only=False)")
    ckpt = torch.load(file, weights_only=False)
    return ckpt, ckpt.get("ema", None)

# Replace the function in ultralytics
ul_tasks.torch_safe_load = patched_safe_load

# Load the model
print("Loading YOLO model...")
try:
    yolo_model = YOLO("yolov8n.pt")
    # Move model to GPU if available
    if torch.cuda.is_available():
        yolo_model.to('cuda')
        print("✅ YOLO model loaded successfully and moved to GPU!")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("✅ YOLO model loaded successfully on CPU!")
except Exception as e:
    print(f"❌ Error loading YOLO model: {e}")
    yolo_model = None

# Global variables for video processing
is_processing = False
current_video_path = None
processing_thread = None

# Global variables for webcam processing
is_webcam_active = False
webcam_thread = None
webcam_cap = None
webcam_frame_buffer = []
webcam_alert_cooldown = {}  # Track last alert time per person to avoid spam

ONNX_MODEL_PATH = "glintr100.onnx"

# Download ONNX model if not present
if ONNX_AVAILABLE and not os.path.exists(ONNX_MODEL_PATH):
    import requests
    print("Downloading ONNX face embedding model...")
    url = "https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcface_r100_v1.onnx"
    r = requests.get(url)
    with open(ONNX_MODEL_PATH, "wb") as f:
        f.write(r.content)
    print("✅ ONNX model downloaded.")

# ONNX face embedding function
def get_face_embedding_onnx(face_img):
    # Preprocess: resize to 112x112, BGR->RGB, normalize
    img = cv2.resize(face_img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
    img = np.transpose(img, (2, 0, 1))[None, ...]  # (1, 3, 112, 112)
    if ONNX_AVAILABLE and os.path.exists(ONNX_MODEL_PATH):
        session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
        embedding = session.run(None, {session.get_inputs()[0].name: img})[0][0]
        return embedding / (np.linalg.norm(embedding) + 1e-8)
    else:
        return None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'mp4', 'avi', 'mov'}

def encode_face_encoding(encoding):
    """Convert numpy array to JSON string"""
    return json.dumps(encoding.tolist())

def decode_face_encoding(encoding_str):
    """Convert JSON string back to numpy array"""
    return np.array(json.loads(encoding_str))

def is_recently_detected_unknown(face_encoding, app_context=None):
    """Check if this face was recently detected as unknown to prevent spam"""
    try:
        if app_context is None:
            with app.app_context():
                return _check_recent_unknowns(face_encoding)
        else:
            return _check_recent_unknowns(face_encoding)
    except:
        return False

def _check_recent_unknowns(face_encoding):
    """Internal function to check recent unknowns"""
    # Get ALL unknown persons (not just recent ones) to remember them
    all_unknowns = UnknownPerson.query.all()
    
    if not all_unknowns:
        return False
    
    # Check similarity with all unknowns
    for unknown in all_unknowns:
        try:
            unknown_encoding = decode_face_encoding(unknown.face_encoding)
            # Calculate cosine similarity
            similarity = np.dot(face_encoding, unknown_encoding) / (
                np.linalg.norm(face_encoding) * np.linalg.norm(unknown_encoding) + 1e-8
            )
            # If very similar (>0.5), consider it the same unknown person
            if similarity > 0.5:
                print(f"🎯 Found matching unknown person (ID: {unknown.id}, Similarity: {similarity:.3f})")
                return True
        except:
            continue
    
    return False

def find_existing_unknown_person(face_encoding):
    """Find if this face matches an existing unknown person"""
    all_unknowns = UnknownPerson.query.all()
    
    best_match = None
    best_similarity = 0.0
    
    for unknown in all_unknowns:
        try:
            unknown_encoding = decode_face_encoding(unknown.face_encoding)
            # Calculate cosine similarity
            similarity = np.dot(face_encoding, unknown_encoding) / (
                np.linalg.norm(face_encoding) * np.linalg.norm(unknown_encoding) + 1e-8
            )
            # Much lower threshold for unknown person matching (0.5 instead of 0.7)
            if similarity > 0.5 and similarity > best_similarity:
                best_match = unknown
                best_similarity = similarity
                print(f"🔍 Found potential match: Unknown ID {unknown.id} with similarity {similarity:.3f}")
        except Exception as e:
            print(f"❌ Error comparing with unknown {unknown.id}: {e}")
            continue
    
    return best_match, best_similarity

def get_face_encodings_opencv(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return [], []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        face_locations = []
        face_encodings = []
        for (x, y, w, h) in faces:
            face_locations.append((y, x + w, y + h, x))
            face_roi = image[y:y+h, x:x+w]
            if ONNX_AVAILABLE:
                embedding = get_face_embedding_onnx(face_roi)
                if embedding is not None:
                    face_encodings.append(embedding)
                    continue
            # Fallback: deterministic histogram
            face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            face_roi_gray = cv2.resize(face_roi_gray, (128, 128))
            hist = cv2.calcHist([face_roi_gray], [0], None, [128], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-8)
            face_encodings.append(hist)
        print(f"OpenCV detected {len(face_locations)} faces")
        return face_encodings, face_locations
    except Exception as e:
        print(f"Error in OpenCV face detection: {e}")
        return [], []

def get_face_encodings_mediapipe(image_path):
    try:
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image from {image_path}")
                return [], []
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_image)
            face_locations = []
            face_encodings = []
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                face_locations.append((y, x + width, y + height, x))
                face_roi = image[y:y+height, x:x+width]
                if ONNX_AVAILABLE:
                    embedding = get_face_embedding_onnx(face_roi)
                    if embedding is not None:
                        face_encodings.append(embedding)
                        continue
                # Fallback: deterministic histogram
                face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                face_roi_gray = cv2.resize(face_roi_gray, (128, 128))
                hist = cv2.calcHist([face_roi_gray], [0], None, [128], [0, 256]).flatten()
                hist = hist / (hist.sum() + 1e-8)
                face_encodings.append(hist)
            print(f"MediaPipe detected {len(face_locations)} faces")
            return face_encodings, face_locations
    except Exception as e:
        print(f"Error in MediaPipe face detection: {e}")
        return [], []

def get_face_encodings_legacy(image_path):
    """Extract face encodings using legacy face_recognition library"""
    try:
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        print(f"Legacy face_recognition detected {len(face_encodings)} faces")
        return face_encodings, face_locations
    except Exception as e:
        print(f"Error in legacy face_recognition: {e}")
        return [], []

def get_face_encodings(image_path):
    """Extract face encodings from an image using available methods"""
    # Try legacy face_recognition first (if available)
    if FACE_RECOGNITION_AVAILABLE:
        face_encodings, face_locations = get_face_encodings_legacy(image_path)
        if len(face_encodings) > 0:
            return face_encodings, face_locations
    
    # Try MediaPipe (preferred modern method)
    if MEDIAPIPE_AVAILABLE:
        print("🔄 Using MediaPipe for face detection...")
        face_encodings, face_locations = get_face_encodings_mediapipe(image_path)
        if len(face_encodings) > 0:
            return face_encodings, face_locations
    
    # Try OpenCV (fallback method)
    if OPENCV_FACE_AVAILABLE:
        print("🔄 Using OpenCV for face detection...")
        face_encodings, face_locations = get_face_encodings_opencv(image_path)
        if len(face_encodings) > 0:
            return face_encodings, face_locations
    
    print("⚠️  No face detection method available - using fallback encoding")
    # Create a fallback encoding based on image features
    try:
        image = cv2.imread(image_path)
        if image is not None:
            # Resize image to consistent size
            image = cv2.resize(image, (128, 128))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Create encoding based on image features
            hist = cv2.calcHist([gray], [0], None, [128], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-8)
            return [hist], [(0, 100, 100, 0)]
    except:
        pass
    
    # Final fallback
    fallback_encoding = np.ones(128)
    return [fallback_encoding], [(0, 100, 100, 0)]

def compare_faces(known_encodings, face_encoding, tolerance=0.6):
    """Compare face encodings and return matches"""
    if not FACE_RECOGNITION_AVAILABLE:
        # Use cosine similarity for alternative methods
        matches = []
        for known_encoding in known_encodings:
            # Calculate cosine similarity
            similarity = np.dot(face_encoding, known_encoding) / (np.linalg.norm(face_encoding) * np.linalg.norm(known_encoding) + 1e-8)
            matches.append(similarity > (1 - tolerance))
        return matches
    
    if len(known_encodings) == 0:
        return []
    
    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
    return matches

def find_best_match(known_encodings, face_encoding, tolerance=0.5):
    """Find the best matching face and return similarity score"""
    if not FACE_RECOGNITION_AVAILABLE:
        # Use cosine similarity for alternative methods
        if len(known_encodings) == 0:
            return None, 0
        
        best_similarity = -1
        best_match_index = None
        
        for i, known_encoding in enumerate(known_encodings):
            # Calculate cosine similarity
            similarity = np.dot(face_encoding, known_encoding) / (np.linalg.norm(face_encoding) * np.linalg.norm(known_encoding) + 1e-8)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_index = i
        
        # Use tolerance parameter for matching
        if best_similarity >= tolerance:
            print(f"✅ Found known person match: Index {best_match_index}, Similarity: {best_similarity:.3f}")
            return best_match_index, best_similarity
        else:
            print(f"❌ No known person match found. Best similarity: {best_similarity:.3f} (threshold: {tolerance})")
        return None, 0
    
    if len(known_encodings) == 0:
        return None, 0
    
    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    best_match_distance = face_distances[best_match_index]
    
    if best_match_distance <= tolerance:
        return best_match_index, 1 - best_match_distance
    return None, 0

def process_video_frames():
    """Process video frames for person detection and face recognition"""
    global is_processing, current_video_path
    
    if not current_video_path or not yolo_model:
        return
    
    cap = cv2.VideoCapture(current_video_path)
    frame_count = 0
    
    while is_processing and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 10th frame to reduce computational load
        if frame_count % 10 == 0:
            # Detect people using YOLO
            results = yolo_model(frame, classes=[0])  # class 0 is person in COCO
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Crop person from frame
                        person_crop = frame[y1:y2, x1:x2]
                        
                        if person_crop.size > 0:
                            # Save temporary image for face detection
                            temp_path = f"temp_person_{frame_count}.jpg"
                            cv2.imwrite(temp_path, person_crop)
                            
                            # Extract face encodings
                            face_encodings, face_locations = get_face_encodings(temp_path)
                            
                            # Clean up temporary file
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            
                            # Use application context for database operations
                            with app.app_context():
                                for face_encoding in face_encodings:
                                    # Get all known face encodings
                                    known_persons = KnownPerson.query.all()
                                    known_encodings = [decode_face_encoding(person.face_encoding) for person in known_persons]
                                    
                                    # Check if face matches known people
                                    best_match_index, similarity = find_best_match(known_encodings, face_encoding)
                                    
                                    if best_match_index is None:
                                        # Unknown person detected - check against all existing unknowns
                                        existing_unknown, similarity_score = find_existing_unknown_person(face_encoding)
                                        
                                        if existing_unknown is None:
                                            # This is a truly new unknown person
                                            unknown_person = UnknownPerson(
                                                face_encoding=encode_face_encoding(face_encoding),
                                                image_path=f"unknown_person_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                                            )
                                            
                                            # Save unknown person image
                                            unknown_image_path = os.path.join(app.config['UNKNOWN_PEOPLE_FOLDER'], unknown_person.image_path)
                                            cv2.imwrite(unknown_image_path, person_crop)
                                            
                                            db.session.add(unknown_person)
                                            db.session.commit()
                                            
                                            # Create alert
                                            alert = Alert(
                                                message=f"🚨 NEW UNKNOWN PERSON DETECTED IN VIDEO at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                                                person_id=unknown_person.id
                                            )
                                            db.session.add(alert)
                                            db.session.commit()
                                            
                                            print(f"🚨 VIDEO ALERT: New unknown person detected! (ID: {unknown_person.id})")
                                        else:
                                            # This is the same unknown person we've seen before
                                            print(f"👤 Same unknown person detected in video (ID: {existing_unknown.id}, Similarity: {similarity_score:.3f})")
                                            
                                            # Update the detection time to show recent activity
                                            existing_unknown.detected_at = datetime.utcnow()
                                            db.session.commit()
                                    else:
                                        known_person = known_persons[best_match_index]
                                        print(f"Known person detected: {known_person.name} (Similarity: {similarity:.3f})")
        
        frame_count += 1
    
    cap.release()

def process_webcam_frames():
    """Process webcam frames for real-time person detection and face recognition"""
    global is_webcam_active, webcam_cap, webcam_frame_buffer, webcam_alert_cooldown
    
    if not webcam_cap or not yolo_model:
        return
    
    frame_count = 0
    last_alert_time = {}  # Track last alert time to prevent spam
    
    while is_webcam_active and webcam_cap.isOpened():
        ret, frame = webcam_cap.read()
        if not ret:
            print("Failed to read webcam frame")
            break
        
        # Process every 5th frame for real-time performance
        if frame_count % 5 == 0:
            # Detect people using YOLO
            results = yolo_model(frame, classes=[0])  # class 0 is person in COCO
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Crop person from frame
                        person_crop = frame[y1:y2, x1:x2]
                        
                        if person_crop.size > 0:
                            # Save temporary image for face detection
                            temp_path = f"temp_webcam_person_{frame_count}.jpg"
                            cv2.imwrite(temp_path, person_crop)
                            
                            # Extract face encodings
                            face_encodings, face_locations = get_face_encodings(temp_path)
                            
                            # Clean up temporary file
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            
                            # Use application context for database operations
                            with app.app_context():
                                for face_encoding in face_encodings:
                                    # Get all known face encodings
                                    known_persons = KnownPerson.query.all()
                                    known_encodings = [decode_face_encoding(person.face_encoding) for person in known_persons]
                                    
                                    # Check if face matches known people
                                    best_match_index, similarity = find_best_match(known_encodings, face_encoding)
                                    
                                    if best_match_index is None:
                                        # Unknown person detected - check against all existing unknowns
                                        current_time = time.time()
                                        
                                        # Create a more stable hash based on face features
                                        face_features = face_encoding[:64]  # Use first 64 features (histogram part)
                                        face_hash = hash(tuple(np.round(face_features * 1000).astype(int)))
                                        
                                        # Check if this face was recently detected (cooldown)
                                        if face_hash not in last_alert_time or (current_time - last_alert_time[face_hash]) > 60:  # 60 second cooldown
                                            # Check if this matches an existing unknown person
                                            existing_unknown, similarity_score = find_existing_unknown_person(face_encoding)
                                            
                                            if existing_unknown is None:
                                                # This is a truly new unknown person
                                                unknown_person = UnknownPerson(
                                                    face_encoding=encode_face_encoding(face_encoding),
                                                    image_path=f"webcam_unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                                                )
                                                
                                                # Save unknown person image
                                                unknown_image_path = os.path.join(app.config['UNKNOWN_PEOPLE_FOLDER'], unknown_person.image_path)
                                                cv2.imwrite(unknown_image_path, person_crop)
                                                
                                                db.session.add(unknown_person)
                                                db.session.commit()
                                                
                                                # Create alert
                                                alert = Alert(
                                                    message=f"🚨 NEW UNKNOWN PERSON DETECTED IN WEBCAM at {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                                    person_id=unknown_person.id
                                                )
                                                db.session.add(alert)
                                                db.session.commit()
                                                
                                                last_alert_time[face_hash] = current_time
                                                print(f"🚨 WEBCAM ALERT: New unknown person detected! (ID: {unknown_person.id})")
                                            else:
                                                # This is the same unknown person we've seen before
                                                print(f"👤 Same unknown person detected again (ID: {existing_unknown.id}, Similarity: {similarity_score:.3f})")
                                                
                                                # Update the detection time to show recent activity
                                                existing_unknown.detected_at = datetime.utcnow()
                                                db.session.commit()
                                                
                                                # Only create alert if we haven't alerted for this person recently
                                                recent_alert = Alert.query.filter(
                                                    Alert.person_id == existing_unknown.id,
                                                    Alert.timestamp >= datetime.utcnow() - timedelta(minutes=30)
                                                ).first()
                                                
                                                if not recent_alert:
                                                    # Create alert for repeated detection
                                                    alert = Alert(
                                                        message=f"⚠️ KNOWN UNKNOWN PERSON DETECTED AGAIN (ID: {existing_unknown.id}) at {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                                                        person_id=existing_unknown.id
                                                    )
                                                    db.session.add(alert)
                                                    db.session.commit()
                                                    print(f"⚠️ Alert created for repeated unknown person detection")
                                                
                                                last_alert_time[face_hash] = current_time
                                        else:
                                            print(f"⏰ Skipping alert - cooldown active for this face")
                                    else:
                                        known_person = known_persons[best_match_index]
                                        print(f"✅ Known person in webcam: {known_person.name} (Similarity: {similarity:.3f})")
        
        frame_count += 1
        time.sleep(0.1)  # Small delay to prevent overwhelming the system
    
    if webcam_cap:
        webcam_cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_known_person', methods=['POST'])
def upload_known_person():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    file = request.files['image']
    name = request.form.get('name', 'Unknown')
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['KNOWN_PEOPLE_FOLDER'], filename)
        file.save(filepath)
        # Extract face encodings (always use ONNX if available)
        image = cv2.imread(filepath)
        face_encodings = []
        face_locations = []
        if ONNX_AVAILABLE:
            # Use OpenCV face detection for registration
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            for (x, y, w, h) in faces:
                face_roi = image[y:y+h, x:x+w]
                embedding = get_face_embedding_onnx(face_roi)
                if embedding is not None:
                    face_encodings.append(embedding)
                    face_locations.append((y, x + w, y + h, x))
        if not face_encodings:
            # Fallback to histogram
            face_encodings, face_locations = get_face_encodings(filepath)
        if len(face_encodings) == 0:
            return jsonify({'error': 'No face detected in the image'}), 400
        face_encoding = face_encodings[0]
        # If a known person with the same name exists, update their embedding and image
        existing = KnownPerson.query.filter_by(name=name).first()
        if existing:
            existing.face_encoding = encode_face_encoding(face_encoding)
            existing.image_path = filename
            db.session.commit()
            person_id = existing.id
        else:
            known_person = KnownPerson(
                name=name,
                face_encoding=encode_face_encoding(face_encoding),
                image_path=filename
            )
            db.session.add(known_person)
            db.session.commit()
            person_id = known_person.id
        return jsonify({
            'message': 'Person added successfully',
            'person_id': person_id,
            'name': name
        })
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global is_processing, current_video_path, processing_thread
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400
    
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Stop any existing processing
        if is_processing:
            is_processing = False
            if processing_thread:
                processing_thread.join()
        
        # Start new processing
        current_video_path = filepath
        is_processing = True
        processing_thread = threading.Thread(target=process_video_frames)
        processing_thread.start()
        
        return jsonify({
            'message': 'Video uploaded and processing started',
            'filename': filename
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    global is_processing
    
    is_processing = False
    return jsonify({'message': 'Processing stopped'})

@app.route('/known_people')
def get_known_people():
    known_people = KnownPerson.query.all()
    return jsonify([{
        'id': person.id,
        'name': person.name,
        'image_path': person.image_path,
        'created_at': person.created_at.isoformat()
    } for person in known_people])

@app.route('/unknown_people')
def get_unknown_people():
    unknown_people = UnknownPerson.query.all()
    return jsonify([{
        'id': person.id,
        'image_path': person.image_path,
        'detected_at': person.detected_at.isoformat(),
        'alert_sent': person.alert_sent
    } for person in unknown_people])

@app.route('/alerts')
def get_alerts():
    alerts = Alert.query.order_by(Alert.timestamp.desc()).limit(50).all()
    return jsonify([{
        'id': alert.id,
        'message': alert.message,
        'timestamp': alert.timestamp.isoformat(),
        'person_id': alert.person_id
    } for alert in alerts])

@app.route('/add_unknown_to_known', methods=['POST'])
def add_unknown_to_known():
    data = request.get_json()
    unknown_id = data.get('unknown_id')
    name = data.get('name')
    
    if not unknown_id or not name:
        return jsonify({'error': 'Missing unknown_id or name'}), 400
    
    unknown_person = UnknownPerson.query.get(unknown_id)
    if not unknown_person:
        return jsonify({'error': 'Unknown person not found'}), 404
    
    # Create known person from unknown person
    known_person = KnownPerson(
        name=name,
        face_encoding=unknown_person.face_encoding,
        image_path=unknown_person.image_path
    )
    
    # Move image from unknown to known folder
    old_path = os.path.join(app.config['UNKNOWN_PEOPLE_FOLDER'], unknown_person.image_path)
    new_path = os.path.join(app.config['KNOWN_PEOPLE_FOLDER'], unknown_person.image_path)
    
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
    
    db.session.add(known_person)
    db.session.delete(unknown_person)
    db.session.commit()
    
    print(f"✅ Added unknown person {unknown_id} to known people as '{name}'")
    
    return jsonify({'message': 'Person added to known people successfully'})

@app.route('/clear_old_unknowns', methods=['POST'])
def clear_old_unknowns():
    """Clear old unknown person entries to prevent database bloat"""
    try:
        # Delete unknown persons older than 24 hours
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        old_unknowns = UnknownPerson.query.filter(
            UnknownPerson.detected_at < cutoff_time
        ).all()
        
        count = len(old_unknowns)
        for unknown in old_unknowns:
            # Delete the image file
            image_path = os.path.join(app.config['UNKNOWN_PEOPLE_FOLDER'], unknown.image_path)
            if os.path.exists(image_path):
                os.remove(image_path)
            db.session.delete(unknown)
        
        db.session.commit()
        print(f"🧹 Cleared {count} old unknown person entries")
        
        return jsonify({'message': f'Cleared {count} old unknown person entries'})
    except Exception as e:
        print(f"❌ Error clearing old unknowns: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/delete_unknown_person/<int:unknown_id>', methods=['DELETE'])
def delete_unknown_person(unknown_id):
    """Delete a specific unknown person"""
    try:
        unknown_person = UnknownPerson.query.get(unknown_id)
        if not unknown_person:
            return jsonify({'error': 'Unknown person not found'}), 404
        
        # Delete the image file
        image_path = os.path.join(app.config['UNKNOWN_PEOPLE_FOLDER'], unknown_person.image_path)
        if os.path.exists(image_path):
            os.remove(image_path)
        
        # Delete from database
        db.session.delete(unknown_person)
        db.session.commit()
        
        print(f"🗑️ Deleted unknown person {unknown_id}")
        return jsonify({'message': 'Unknown person deleted successfully'})
    except Exception as e:
        print(f"❌ Error deleting unknown person: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/images/<folder>/<filename>')
def serve_image(folder, filename):
    if folder == 'known':
        return send_from_directory(app.config['KNOWN_PEOPLE_FOLDER'], filename)
    elif folder == 'unknown':
        return send_from_directory(app.config['UNKNOWN_PEOPLE_FOLDER'], filename)
    else:
        return jsonify({'error': 'Invalid folder'}), 400

# Add WebRTC imports and setup
import asyncio
import json
from flask_socketio import SocketIO, emit, join_room, leave_room
import cv2
import numpy as np
import base64
import threading
import time
from collections import deque

# Initialize SocketIO for WebRTC signaling
socketio = SocketIO(app, cors_allowed_origins="*")

# WebRTC global variables
webrtc_connections = {}
webrtc_frame_buffer = deque(maxlen=30)  # Buffer last 30 frames
webrtc_active = False
webrtc_thread = None

def webrtc_frame_generator():
    """Generate frames for WebRTC streaming"""
    global webrtc_active, webcam_cap, webrtc_frame_buffer
    
    print("🎥 Starting WebRTC frame generator...")
    
    while webrtc_active and webcam_cap and webcam_cap.isOpened():
        ret, frame = webcam_cap.read()
        if not ret:
            print("❌ Failed to read webcam frame for WebRTC")
            time.sleep(0.1)
            continue
        
        try:
            # Process frame with YOLO detection if available
            display_frame = frame.copy()
            
            if yolo_model:
                # Detect people using YOLO
                results = yolo_model(frame, classes=[0])  # class 0 is person in COCO
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            # Get confidence score
                            conf = float(box.conf[0].cpu().numpy())
                            
                            # Crop person from frame for face detection
                            person_crop = frame[y1:y2, x1:x2]
                            
                            if person_crop.size > 0:
                                # Save temporary image for face detection
                                temp_path = f"temp_webrtc_person_{int(time.time() * 1000)}.jpg"
                                cv2.imwrite(temp_path, person_crop)
                                
                                # Extract face encodings
                                face_encodings, face_locations = get_face_encodings(temp_path)
                                
                                # Clean up temporary file
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                                
                                # Determine if person is known or unknown
                                person_status = "Unknown"
                                person_name = "Unknown"
                                box_color = (0, 0, 255)  # Red for unknown
                                
                                # Use application context for database operations
                                with app.app_context():
                                    for face_encoding in face_encodings:
                                        # Get all known face encodings
                                        known_persons = KnownPerson.query.all()
                                        known_encodings = [decode_face_encoding(person.face_encoding) for person in known_persons]
                                        
                                        # Check if face matches known people (using 0.5 threshold)
                                        best_match_index, similarity = find_best_match(known_encodings, face_encoding)
                                        
                                        if best_match_index is not None:
                                            known_person = known_persons[best_match_index]
                                            person_status = "Known"
                                            person_name = known_person.name
                                            box_color = (0, 255, 0)  # Green for known
                                            break
                                
                                # Draw bounding box
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
                                
                                # Draw label background
                                label = f"{person_name} ({conf:.2f})"
                                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                                            (x1 + label_size[0], y1), box_color, -1)
                                
                                # Draw label text
                                cv2.putText(display_frame, label, (x1, y1 - 5), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                
                                # Add status indicator
                                status_text = "✅" if person_status == "Known" else "❌"
                                cv2.putText(display_frame, status_text, (x2 - 30, y1 + 20), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
            
            # Resize frame for WebRTC display (maintain aspect ratio)
            # Keep high quality for processing, resize for display
            display_frame = cv2.resize(display_frame, (960, 540))  # 16:9 aspect ratio, good quality
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                # Convert to base64
                frame_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
                
                # Add to buffer
                webrtc_frame_buffer.append({
                    'frame': frame_base64,
                    'timestamp': time.time()
                })
                
                # Broadcast to all connected WebRTC clients
                socketio.emit('webrtc_frame', {
                    'frame': frame_base64,
                    'timestamp': time.time()
                })
            
        except Exception as e:
            print(f"❌ Error processing WebRTC frame: {e}")
        
        time.sleep(0.1)  # 10 FPS
    
    print("🛑 WebRTC frame generator stopped")

# WebRTC Socket.IO event handlers
@socketio.on('connect')
def handle_connect():
    print(f"🔗 WebRTC client connected: {request.sid}")
    webrtc_connections[request.sid] = {
        'connected': True,
        'room': 'webrtc_room'
    }
    join_room('webrtc_room')
    emit('webrtc_connected', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f"🔌 WebRTC client disconnected: {request.sid}")
    if request.sid in webrtc_connections:
        del webrtc_connections[request.sid]
    leave_room('webrtc_room')

@socketio.on('webrtc_offer')
def handle_webrtc_offer(data):
    """Handle WebRTC offer from client"""
    print(f"📤 Received WebRTC offer from {request.sid}")
    # For now, we'll use a simple approach with base64 frames
    # In a full implementation, you'd handle actual WebRTC signaling here
    emit('webrtc_answer', {
        'type': 'answer',
        'sdp': 'dummy-sdp',  # Placeholder
        'status': 'ready'
    }, room=request.sid)

@socketio.on('request_frame')
def handle_request_frame():
    """Handle frame request from client"""
    if webrtc_frame_buffer:
        latest_frame = webrtc_frame_buffer[-1]
        emit('webrtc_frame', latest_frame, room=request.sid)

# Update the start_webcam function to also start WebRTC
@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    """Start webcam for real-time detection"""
    global is_webcam_active, webcam_thread, webcam_cap, webrtc_active, webrtc_thread
    
    print("🚀 Starting webcam...")
    
    if is_webcam_active:
        print("⚠️ Webcam is already active")
        return jsonify({'error': 'Webcam is already active'}), 400
    
    try:
        # Try to open webcam (usually index 0)
        print("📹 Trying camera index 0...")
        webcam_cap = cv2.VideoCapture(0)
        
        if not webcam_cap.isOpened():
            print("❌ Camera index 0 failed, trying alternatives...")
            # Try alternative camera indices
            for i in range(1, 5):
                print(f"📹 Trying camera index {i}...")
                webcam_cap = cv2.VideoCapture(i)
                if webcam_cap.isOpened():
                    print(f"✅ Camera index {i} opened successfully")
                    break
                else:
                    print(f"❌ Camera index {i} failed")
        
        if not webcam_cap.isOpened():
            print("❌ All camera indices failed")
            return jsonify({'error': 'Could not open webcam - no camera found'}), 500
        
        print("✅ Webcam opened successfully")
        
        # Set webcam properties for high quality (720p, 30fps)
        print("⚙️ Setting webcam properties for 720p quality...")
        webcam_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 720p width
        webcam_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 720p height
        webcam_cap.set(cv2.CAP_PROP_FPS, 30)  # 30fps
        webcam_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        # Test if we can actually read a frame
        print("🧪 Testing frame capture...")
        ret, test_frame = webcam_cap.read()
        if not ret:
            print("❌ Failed to read test frame")
            webcam_cap.release()
            return jsonify({'error': 'Webcam opened but cannot read frames'}), 500
        
        print(f"✅ Test frame captured successfully: {test_frame.shape}")
        
        is_webcam_active = True
        webrtc_active = True
        
        print("🔄 Starting webcam processing thread...")
        webcam_thread = threading.Thread(target=process_webcam_frames)
        webcam_thread.daemon = True
        webcam_thread.start()
        
        print("🔄 Starting WebRTC frame generator...")
        webrtc_thread = threading.Thread(target=webrtc_frame_generator)
        webrtc_thread.daemon = True
        webrtc_thread.start()
        
        print("✅ Webcam and WebRTC started successfully")
        return jsonify({
            'message': 'Webcam and WebRTC started successfully',
            'status': 'active',
            'webrtc_support': True
        })
        
    except Exception as e:
        print(f"❌ Error starting webcam: {str(e)}")
        if webcam_cap:
            webcam_cap.release()
            webcam_cap = None
        return jsonify({'error': f'Failed to start webcam: {str(e)}'}), 500

# Update the stop_webcam function
@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    """Stop webcam processing"""
    global is_webcam_active, webcam_cap, webrtc_active
    
    is_webcam_active = False
    webrtc_active = False
    
    if webcam_cap:
        webcam_cap.release()
        webcam_cap = None
    
    return jsonify({
        'message': 'Webcam and WebRTC stopped successfully',
        'status': 'inactive'
    })

# Add WebRTC status endpoint
@app.route('/webrtc_status')
def get_webrtc_status():
    """Get WebRTC status"""
    return jsonify({
        'webrtc_active': webrtc_active,
        'connected_clients': len(webrtc_connections),
        'frame_buffer_size': len(webrtc_frame_buffer),
        'webcam_active': is_webcam_active
    })

# Add WebRTC frame endpoint (fallback for non-WebSocket clients)
@app.route('/webrtc_frame')
def get_webrtc_frame():
    """Get latest WebRTC frame (fallback endpoint)"""
    if webrtc_frame_buffer:
        latest_frame = webrtc_frame_buffer[-1]
        return jsonify(latest_frame)
    else:
        return jsonify({'error': 'No frames available'}), 404

@app.route('/webcam_status')
def get_webcam_status():
    """Get current webcam status"""
    return jsonify({
        'is_active': is_webcam_active,
        'camera_available': webcam_cap is not None and webcam_cap.isOpened() if webcam_cap else False,
        'webrtc_active': webrtc_active,
        'webrtc_connections': len(webrtc_connections)
    })

@app.route('/status')
def get_status():
    return jsonify({
        'is_processing': is_processing,
        'current_video': current_video_path,
        'webcam_active': is_webcam_active,
        'face_detection_methods': {
            'legacy_face_recognition': FACE_RECOGNITION_AVAILABLE,
            'mediapipe': MEDIAPIPE_AVAILABLE,
            'opencv': OPENCV_FACE_AVAILABLE
        },
        'yolo_available': yolo_model is not None
    })

@app.route('/system_info')
def get_system_info():
    """Get system information and feature availability"""
    # Determine the best available face detection method
    face_detection_method = "none"
    if FACE_RECOGNITION_AVAILABLE:
        face_detection_method = "legacy_face_recognition"
    elif MEDIAPIPE_AVAILABLE:
        face_detection_method = "mediapipe"
    elif OPENCV_FACE_AVAILABLE:
        face_detection_method = "opencv"
    
    # Get GPU information
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A'
    }
    
    return jsonify({
        'face_detection_methods': {
            'legacy_face_recognition': FACE_RECOGNITION_AVAILABLE,
            'mediapipe': MEDIAPIPE_AVAILABLE,
            'opencv': OPENCV_FACE_AVAILABLE
        },
        'best_face_detection_method': face_detection_method,
        'yolo_available': yolo_model is not None,
        'gpu_info': gpu_info,
        'features': {
            'person_detection': yolo_model is not None,
            'face_detection': face_detection_method != "none",
            'face_recognition': FACE_RECOGNITION_AVAILABLE,  # Only true if legacy method is available
            'video_processing': True,
            'webcam_processing': True,
            'real_time_alerts': True,
            'web_interface': True,
            'gpu_acceleration': torch.cuda.is_available()
        }
    })

@app.route('/webcam_test')
def webcam_test():
    """Test webcam by capturing a single frame"""
    global webcam_cap, is_webcam_active
    
    print(f"🧪 Webcam test requested - Active: {is_webcam_active}, Cap: {webcam_cap is not None}")
    
    # If webcam is not active, try to start it temporarily
    if not is_webcam_active or not webcam_cap or not webcam_cap.isOpened():
        print("🔄 Webcam not active, attempting to start temporarily...")
        try:
            # Try to open webcam (usually index 0)
            temp_cap = cv2.VideoCapture(0)
            
            if not temp_cap.isOpened():
                # Try alternative camera indices
                for i in range(1, 5):
                    print(f"📹 Trying camera index {i}...")
                    temp_cap = cv2.VideoCapture(i)
                    if temp_cap.isOpened():
                        print(f"✅ Camera index {i} opened successfully")
                        break
            
            if not temp_cap.isOpened():
                return jsonify({'error': 'No camera available'}), 400
            
            # Set camera properties for high quality
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 720p width
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 720p height
            temp_cap.set(cv2.CAP_PROP_FPS, 30)  # 30fps
            
            # Capture a test frame
            ret, frame = temp_cap.read()
            if not ret:
                temp_cap.release()
                return jsonify({'error': 'Failed to read webcam frame'}), 500
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                temp_cap.release()
                return jsonify({'error': 'Failed to encode frame'}), 500
            
            # Convert to base64
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Release temporary camera
            temp_cap.release()
            
            return jsonify({
                'success': True,
                'frame': f"data:image/jpeg;base64,{frame_base64}",
                'webcam_active': False,
                'cap_opened': False,
                'message': 'Camera test successful - camera is available but not actively streaming'
            })
            
        except Exception as e:
            print(f"❌ Error in temporary webcam test: {e}")
            return jsonify({'error': f'Camera test failed: {str(e)}'}), 500
    
    # If webcam is already active, use it
    try:
        ret, frame = webcam_cap.read()
        if not ret:
            return jsonify({'error': 'Failed to read webcam frame'}), 500
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            return jsonify({'error': 'Failed to encode frame'}), 500
        
        # Convert to base64
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'frame': f"data:image/jpeg;base64,{frame_base64}",
            'webcam_active': is_webcam_active,
            'cap_opened': webcam_cap.isOpened()
        })
        
    except Exception as e:
        print(f"❌ Error in webcam test: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect_cameras')
def detect_cameras():
    """Detect available cameras on the system"""
    available_cameras = []
    
    print("🔍 Detecting available cameras...")
    
    # Test camera indices 0-9
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to read a frame to confirm it's working
                ret, frame = cap.read()
                if ret:
                    available_cameras.append({
                        'index': i,
                        'resolution': f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
                        'fps': cap.get(cv2.CAP_PROP_FPS),
                        'working': True
                    })
                    print(f"✅ Camera {i} found and working")
                else:
                    available_cameras.append({
                        'index': i,
                        'resolution': 'Unknown',
                        'fps': 0,
                        'working': False
                    })
                    print(f"⚠️ Camera {i} found but cannot read frames")
                cap.release()
            else:
                print(f"❌ Camera {i} not available")
        except Exception as e:
            print(f"❌ Error testing camera {i}: {e}")
    
    print(f"📊 Found {len(available_cameras)} available cameras")
    
    return jsonify({
        'available_cameras': available_cameras,
        'total_cameras': len(available_cameras),
        'current_webcam_status': {
            'is_active': is_webcam_active,
            'cap_exists': webcam_cap is not None,
            'cap_opened': webcam_cap.isOpened() if webcam_cap else False
        }
    })

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000) 