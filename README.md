# 🛡️ Surveillance System - Person Detection & Recognition

A comprehensive web application that uses YOLO for person detection and face recognition to identify known and unknown individuals in video feeds. The system provides real-time alerts for unknown persons and allows you to manage a database of known people.

## ✨ Features

### 🔍 Person Detection
- **YOLOv8 Integration**: Uses YOLOv8 for accurate person detection in video frames
- **Real-time Processing**: Processes video frames to detect people
- **Bounding Box Detection**: Identifies person locations with precise bounding boxes

### 👤 Face Recognition
- **Face Extraction**: Automatically extracts faces from detected persons
- **Face Encoding**: Generates 128-dimensional face embeddings using `face_recognition` library
- **Similarity Matching**: Compares detected faces with known people database

### 📊 Known People Management
- **Add Known People**: Upload photos and add names to the known people database
- **Face Database**: Stores face encodings for quick comparison
- **Person Profiles**: View all known people with their photos and details

### 🚨 Alert System
- **Unknown Person Detection**: Automatically detects and alerts for unknown individuals
- **Real-time Alerts**: Instant notifications when unknown persons are detected
- **Alert History**: View all past alerts with timestamps

### 🎯 Unknown Person Management
- **Detection Log**: View all unknown person detections
- **Add to Known**: Easily add unknown persons to the known people database
- **Image Storage**: Automatically saves images of detected unknown persons

### 🎨 Modern Web Interface
- **Responsive Design**: Beautiful, modern UI that works on all devices
- **Real-time Updates**: Live dashboard with automatic updates
- **Drag & Drop**: Easy file upload with drag and drop support
- **Status Monitoring**: Real-time processing status indicators

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Windows 10/11 (tested on Windows)
- Webcam or video files for testing

### Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd surveillance
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the web interface**
   - Open your browser and go to: `http://localhost:5000`
   - The application will automatically create necessary directories and database

## 📋 System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space for models and database
- **GPU**: Optional but recommended for faster processing

### Software Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows 10/11, Linux, or macOS
- **Browser**: Modern web browser (Chrome, Firefox, Safari, Edge)

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root (optional):
```env
FLASK_SECRET_KEY=your-secret-key-here
FLASK_ENV=development
```

### Model Configuration
The system uses YOLOv8 by default. You can modify the model in `app.py`:
```python
# Change model size for different speed/accuracy trade-offs
yolo_model = YOLO('yolov8n.pt')  # nano (fastest)
# yolo_model = YOLO('yolov8s.pt')  # small
# yolo_model = YOLO('yolov8m.pt')  # medium
# yolo_model = YOLO('yolov8l.pt')  # large
# yolo_model = YOLO('yolov8x.pt')  # extra large (most accurate)
```

### Face Recognition Settings
Adjust face recognition sensitivity in `app.py`:
```python
# Tolerance for face matching (lower = more strict)
tolerance = 0.6  # Default: 0.6 (60% similarity required)
```

## 📖 Usage Guide

### 1. Adding Known People

1. **Navigate to "Add Known Person" section**
2. **Enter the person's name** in the text field
3. **Upload a clear photo** of the person's face
   - Drag and drop an image file or click "Choose Image"
   - Ensure the face is clearly visible and well-lit
4. **Click "Add Person"** to save to the database

### 2. Processing Videos

1. **Navigate to "Video Processing" section**
2. **Upload a video file**
   - Supported formats: MP4, AVI, MOV
   - Drag and drop or click "Choose Video"
3. **Processing starts automatically**
   - The system will analyze each frame for people
   - Known people are identified and logged
   - Unknown people trigger alerts

### 3. Managing Unknown Detections

1. **View unknown detections** in the "Unknown Detections" section
2. **Review detected images** to identify the person
3. **Click "Add"** next to any unknown person
4. **Enter their name** in the modal dialog
5. **Confirm** to add them to known people

### 4. Monitoring Alerts

- **Real-time alerts** appear in the "Recent Alerts" section
- **Alert details** include timestamp and detection information
- **Alert history** is automatically maintained

## 🗂️ Project Structure

```
surveillance/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Main web interface
├── uploads/              # Uploaded video files
├── known_people/         # Known person images
├── unknown_people/       # Unknown person images
└── surveillance.db       # SQLite database (auto-created)
```

## 🔍 API Endpoints

### Core Endpoints
- `GET /` - Main dashboard
- `GET /status` - Processing status
- `GET /known_people` - List known people
- `GET /unknown_people` - List unknown detections
- `GET /alerts` - List recent alerts

### Upload Endpoints
- `POST /upload_video` - Upload video for processing
- `POST /upload_known_person` - Add known person
- `POST /stop_processing` - Stop video processing

### Management Endpoints
- `POST /add_unknown_to_known` - Add unknown person to known database
- `GET /images/<folder>/<filename>` - Serve images

## 🛠️ Troubleshooting

### Common Issues

1. **Installation Problems**
   ```bash
   # If you get dlib installation errors on Windows:
   pip install cmake
   pip install dlib
   ```

2. **CUDA/GPU Issues**
   ```bash
   # Install CPU-only PyTorch if GPU issues occur:
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Memory Issues**
   - Reduce video resolution
   - Use smaller YOLO model (yolov8n.pt)
   - Process shorter video segments

4. **Face Detection Issues**
   - Ensure good lighting in images
   - Use clear, front-facing photos
   - Check that faces are not too small in images

### Performance Optimization

1. **Faster Processing**
   - Use YOLOv8n (nano) model
   - Reduce video resolution
   - Process fewer frames (increase frame skip)

2. **Better Accuracy**
   - Use YOLOv8x (extra large) model
   - Improve image quality
   - Use multiple photos per person

## 🔒 Security Considerations

- **Local Processing**: All processing happens locally on your machine
- **No Cloud Dependencies**: No data is sent to external services
- **Secure Storage**: Images and data stored locally
- **Access Control**: Consider adding authentication for production use

## 🚀 Deployment

### Production Deployment

1. **Use a production WSGI server**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

2. **Add authentication**
   - Implement user login system
   - Add role-based access control
   - Secure API endpoints

3. **Database optimization**
   - Use PostgreSQL for larger datasets
   - Implement database indexing
   - Regular database maintenance

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for object detection
- **face_recognition**: Adam Geitgey for face recognition library
- **Flask**: Web framework
- **Bootstrap**: UI components

## 📞 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section
- Review the documentation

---

**Note**: This system is designed for educational and personal use. Ensure compliance with local privacy laws and regulations when deploying in production environments. 