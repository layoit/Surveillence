import os
from datetime import datetime

class Config:
    """Configuration class for the surveillance system"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-this-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Database Configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///surveillance.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # File Storage Configuration
    UPLOAD_FOLDER = 'uploads'
    KNOWN_PEOPLE_FOLDER = 'known_people'
    UNKNOWN_PEOPLE_FOLDER = 'unknown_people'
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    
    # Allowed file extensions
    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'}
    
    # YOLO Configuration
    YOLO_MODEL = 'yolov8n.pt'  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    YOLO_CONFIDENCE = 0.5  # Minimum confidence for person detection
    YOLO_CLASSES = [0]  # Class 0 is person in COCO dataset
    
    # Face Detection Configuration
    FACE_DETECTION_TOLERANCE = 0.6  # Lower = more strict matching (0.4-0.7 recommended)
    FACE_DETECTION_METHOD = 'auto'  # Options: 'auto', 'mediapipe', 'opencv', 'none'
    
    # MediaPipe Configuration
    MEDIAPIPE_DETECTION_CONFIDENCE = 0.5  # Minimum confidence for MediaPipe face detection
    MEDIAPIPE_MODEL_SELECTION = 1  # 0 for short-range, 1 for full-range
    
    # OpenCV Face Detection Configuration
    OPENCV_SCALE_FACTOR = 1.1  # Scale factor for OpenCV face detection
    OPENCV_MIN_NEIGHBORS = 4  # Minimum neighbors for OpenCV face detection
    
    # Video Processing Configuration
    FRAME_SKIP = 10  # Process every Nth frame (higher = faster, lower = more accurate)
    MAX_FRAME_SIZE = (640, 480)  # Resize frames for processing (width, height)
    
    # Alert Configuration
    ALERT_RETENTION_DAYS = 30  # How long to keep alerts in database
    MAX_ALERTS_DISPLAY = 50  # Maximum alerts to show in dashboard
    
    # Performance Configuration
    THREADING_ENABLED = True  # Enable multi-threading for video processing
    GPU_ENABLED = True  # Enable GPU acceleration if available
    
    # Security Configuration
    ENABLE_CORS = True  # Enable Cross-Origin Resource Sharing
    RATE_LIMIT_ENABLED = False  # Enable rate limiting for API endpoints
    
    @staticmethod
    def get_allowed_extensions():
        """Get all allowed file extensions"""
        return Config.ALLOWED_IMAGE_EXTENSIONS.union(Config.ALLOWED_VIDEO_EXTENSIONS)
    
    @staticmethod
    def is_image_file(filename):
        """Check if file is an allowed image type"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_IMAGE_EXTENSIONS
    
    @staticmethod
    def is_video_file(filename):
        """Check if file is an allowed video type"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_VIDEO_EXTENSIONS
    
    @staticmethod
    def get_timestamp():
        """Get current timestamp string"""
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    @staticmethod
    def get_formatted_datetime():
        """Get current formatted datetime string"""
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    THREADING_ENABLED = True
    GPU_ENABLED = False  # Disable GPU in development to avoid conflicts

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(24)
    THREADING_ENABLED = True
    GPU_ENABLED = True
    RATE_LIMIT_ENABLED = True

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    THREADING_ENABLED = False
    GPU_ENABLED = False

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 