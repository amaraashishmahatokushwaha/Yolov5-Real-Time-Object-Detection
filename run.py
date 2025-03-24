from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import os
import threading
import torch
import socket
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
camera = None
output_frame = None
lock = threading.Lock()
detection_enabled = True
model = None

# Load YOLOv5 model
def load_model():
    global model
    try:
        # Load YOLOv5 model
        logger.info("Loading YOLOv5 model...")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
        # Configure model
        model.conf = 0.45  # Confidence threshold
        model.iou = 0.45   # IoU threshold
        model.classes = None  # All classes
        model.max_det = 50  # Maximum detections
        logger.info("YOLOv5 model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

# Ensure model file exists
def check_model_file():
    if not os.path.exists('yolov5s.pt'):
        logger.error("Missing yolov5s.pt file")
        return False
    logger.info("Found yolov5s.pt model file")
    return True

# Perform object detection on a frame
def detect_objects(frame):
    global model
    
    if model is None:
        model = load_model()
        if model is None:
            # If model failed to load, return original frame
            return frame
    
    try:
        # Convert BGR to RGB (YOLOv5 expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform inference
        results = model(rgb_frame)
        
        # Get detections and draw them on the frame
        detections = results.pandas().xyxy[0]
        
        # Draw bounding boxes on the frame
        for _, detection in detections.iterrows():
            x1, y1 = int(detection['xmin']), int(detection['ymin'])
            x2, y2 = int(detection['xmax']), int(detection['ymax'])
            label = detection['name']
            confidence = detection['confidence']
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
    
    return frame

# Function to capture frames from webcam
def capture_frames():
    global camera, output_frame, lock, detection_enabled
    
    # Check if model file exists
    if not check_model_file():
        logger.error("Cannot start camera: missing model file")
        return
    
    # Load YOLOv5 model
    if model is None:
        load_model()
    
    logger.info("Starting frame capture loop")
    frame_count = 0
    
    while True:
        if camera is None:
            logger.info("Camera object is None, exiting capture loop")
            break
            
        success, frame = camera.read()
        if not success:
            logger.warning("Failed to read frame, retrying...")
            # Try to reconnect if the camera is still available
            if camera is not None:
                continue
            else:
                break
            
        # Apply object detection if enabled (only on every other frame to improve performance)
        if detection_enabled:
            frame = detect_objects(frame)
            
        # Add a status on the frame
        cv2.putText(
            frame, 
            f"Detection: {'ON' if detection_enabled else 'OFF'}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 0, 255), 
            2
        )
        
        # Add IP address to frame
        cv2.putText(
            frame,
            f"Access: http://{get_ip_address()}:5000",
            (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )
            
        # Update the output frame
        with lock:
            output_frame = frame.copy()
            
        frame_count += 1
        if frame_count % 100 == 0:
            logger.info(f"Processed {frame_count} frames")

# Generate video frames for streaming
def generate():
    global output_frame, lock
    
    while True:
        with lock:
            if output_frame is None:
                continue
            
            # Encode the frame as JPEG
            try:
                (flag, encoded_frame) = cv2.imencode(".jpg", output_frame)
                
                if not flag:
                    continue
            except Exception as e:
                logger.error(f"Error encoding frame: {str(e)}")
                continue
                
        # Yield the output frame in byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encoded_frame) + b'\r\n')

# Get the IP address of the machine
def get_ip_address():
    try:
        # Get the primary IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logger.error(f"Error getting IP address: {str(e)}")
        return "127.0.0.1"  # Fallback to localhost

# API routes
@app.route('/')
def index():
    ip_address = get_ip_address()
    return render_template('index.html', ip_address=ip_address)

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera, detection_enabled
    
    if camera is not None:
        return jsonify({"status": "Camera is already running"})
    
    try:
        # Open the laptop's webcam (0 is usually the default camera)
        logger.info("Opening camera...")
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            logger.error("Failed to open camera")
            return jsonify({"status": "Failed to open camera", "success": False})
        
        # Start the camera thread
        detection_enabled = request.json.get('detection', True) if request.is_json else True
        threading.Thread(target=capture_frames, daemon=True).start()
        logger.info("Camera started successfully")
        
        return jsonify({"status": "Camera started successfully", "success": True})
    except Exception as e:
        logger.error(f"Error starting camera: {str(e)}")
        return jsonify({"status": f"Error: {str(e)}", "success": False})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera, output_frame
    
    if camera is None:
        return jsonify({"status": "Camera is not running", "success": False})
    
    try:
        # Release the camera
        logger.info("Stopping camera...")
        camera.release()
        camera = None
        
        # Clear the output frame
        with lock:
            output_frame = None
        
        logger.info("Camera stopped successfully")
        return jsonify({"status": "Camera stopped successfully", "success": True})
    except Exception as e:
        logger.error(f"Error stopping camera: {str(e)}")
        return jsonify({"status": f"Error: {str(e)}", "success": False})

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global detection_enabled
    
    if camera is None:
        return jsonify({"status": "Camera is not running", "success": False})
    
    try:
        detection_enabled = not detection_enabled
        logger.info(f"Detection toggled: {detection_enabled}")
        
        return jsonify({
            "status": "Detection toggled successfully", 
            "detection_enabled": detection_enabled,
            "success": True
        })
    except Exception as e:
        logger.error(f"Error toggling detection: {str(e)}")
        return jsonify({"status": f"Error: {str(e)}", "success": False})

@app.route('/status', methods=['GET'])
def get_status():
    global camera, detection_enabled
    
    camera_running = camera is not None
    
    return jsonify({
        "camera_running": camera_running,
        "detection_enabled": detection_enabled if camera_running else False,
        "server_ip": get_ip_address()
    })

# Create HTML template directory and file
def create_template():
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>YOLOv5 Network Object Detection</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 100%;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 20px;
            font-size: 14px;
        }
        .video-container {
            margin: 20px auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow: hidden;
            max-width: 800px;
            position: relative;
        }
        .video-container img {
            width: 100%;
            height: auto;
            display: block;
        }
        .controls {
            display: flex;
            justify-content: center;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        button {
            padding: 12px 24px;
            margin: 5px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s;
            font-weight: bold;
        }
        .start {
            background-color: #4CAF50;
            color: white;
        }
        .stop {
            background-color: #f44336;
            color: white;
        }
        .toggle {
            background-color: #2196F3;
            color: white;
        }
        button:hover {
            opacity: 0.8;
            transform: scale(1.05);
        }
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        .status {
            text-align: center;
            margin: 15px 0;
            padding: 10px;
            border-radius: 5px;
            font-weight: bold;
            background-color: #f0f0f0;
        }
        .status.running {
            background-color: #e8f5e9;
            color: #4CAF50;
        }
        .status.stopped {
            background-color: #ffebee;
            color: #f44336;
        }
        .network-info {
            text-align: center;
            margin-top: 20px;
            padding: 10px;
            background-color: #e3f2fd;
            border-radius: 5px;
        }
        .loading {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: rgba(0,0,0,0.5);
            display: flex;
            justify-content: center;
            align-items: center;
            color: white;
            font-size: 20px;
        }
        .spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #3498db;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 2s linear infinite;
            margin-right: 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        @media (max-width: 600px) {
            .container {
                padding: 10px;
            }
            button {
                padding: 10px 20px;
                font-size: 14px;
                width: 120px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>YOLOv5 Object Detection</h1>
        <div class="subtitle">Access from any device on the network at: <span id="server-address">http://{{ ip_address }}:5000</span></div>
        
        <div class="controls">
            <button class="start" id="startBtn" onclick="startCamera()">Start Camera</button>
            <button class="stop" id="stopBtn" onclick="stopCamera()" disabled>Stop Camera</button>
            <button class="toggle" id="toggleBtn" onclick="toggleDetection()" disabled>Toggle Detection</button>
        </div>
        
        <div class="status" id="statusBox">Camera is stopped. Click "Start Camera" to begin.</div>
        
        <div class="video-container">
            <img id="video" src="{{ url_for('video_feed') }}" alt="Video feed will appear here when camera is started" style="display: none;">
            <div id="videoPlaceholder">Video feed will appear here when camera is started</div>
            <div class="loading" id="loadingIndicator" style="display: none;">
                <div class="spinner"></div>
                <div>Loading...</div>
            </div>
        </div>
        
        <div class="network-info">
            <p>To access this camera from another device on the same network:</p>
            <p>Open a web browser and navigate to: <strong id="networkUrl">http://{{ ip_address }}:5000</strong></p>
        </div>
    </div>

    <script>
        let cameraRunning = false;
        let detectionEnabled = true;
        
        // Update UI based on current status
        function updateUI() {
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const toggleBtn = document.getElementById('toggleBtn');
            const statusBox = document.getElementById('statusBox');
            const video = document.getElementById('video');
            const videoPlaceholder = document.getElementById('videoPlaceholder');
            
            startBtn.disabled = cameraRunning;
            stopBtn.disabled = !cameraRunning;
            toggleBtn.disabled = !cameraRunning;
            
            if (cameraRunning) {
                statusBox.textContent = `Camera is running. Detection is ${detectionEnabled ? 'ON' : 'OFF'}.`;
                statusBox.className = 'status running';
                video.style.display = 'block';
                videoPlaceholder.style.display = 'none';
            } else {
                statusBox.textContent = 'Camera is stopped. Click "Start Camera" to begin.';
                statusBox.className = 'status stopped';
                video.style.display = 'none';
                videoPlaceholder.style.display = 'block';
            }
        }
        
        // Function to start the camera
        function startCamera() {
            showLoading(true);
            fetch('/start_camera', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({detection: true})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    cameraRunning = true;
                    detectionEnabled = true;
                    // Refresh the video source to ensure it connects to the stream
                    document.getElementById('video').src = "{{ url_for('video_feed') }}?" + new Date().getTime();
                } else {
                    alert('Error: ' + data.status);
                }
                updateUI();
                showLoading(false);
            })
            .catch(error => {
                console.error('Error:', error);
                showLoading(false);
                alert('Error starting camera: ' + error);
            });
        }
        
        // Function to stop the camera
        function stopCamera() {
            showLoading(true);
            fetch('/stop_camera', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    cameraRunning = false;
                }
                updateUI();
                showLoading(false);
            })
            .catch(error => {
                console.error('Error:', error);
                showLoading(false);
                alert('Error stopping camera: ' + error);
            });
        }
        
        // Function to toggle detection
        function toggleDetection() {
            if (!cameraRunning) return;
            
            fetch('/toggle_detection', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    detectionEnabled = data.detection_enabled;
                    updateUI();
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error toggling detection: ' + error);
            });
        }
        
        // Function to show/hide loading indicator
        function showLoading(show) {
            document.getElementById('loadingIndicator').style.display = show ? 'flex' : 'none';
        }
        
        // Function to get current status
        function getStatus() {
            fetch('/status')
            .then(response => response.json())
            .then(data => {
                cameraRunning = data.camera_running;
                detectionEnabled = data.detection_enabled;
                
                // Update server IP display
                const serverAddressElements = document.querySelectorAll('#server-address, #networkUrl');
                serverAddressElements.forEach(el => {
                    el.textContent = `http://${data.server_ip}:5000`;
                });
                
                updateUI();
                
                if (cameraRunning) {
                    // Ensure video is properly loaded
                    document.getElementById('video').src = "{{ url_for('video_feed') }}?" + new Date().getTime();
                }
            })
            .catch(error => {
                console.error('Error fetching status:', error);
            });
        }
        
        // Initialize on page load
        window.onload = function() {
            getStatus();
            
            // Add event listener for video load errors
            document.getElementById('video').onerror = function() {
                if (cameraRunning) {
                    console.log('Video stream error, attempting to reconnect...');
                    setTimeout(() => {
                        this.src = "{{ url_for('video_feed') }}?" + new Date().getTime();
                    }, 2000);
                }
            };
        };
    </script>
</body>
</html>
        ''')

if __name__ == '__main__':
    create_template()
    ip_address = get_ip_address()
    logger.info("Starting YOLOv5 object detection server...")
    logger.info(f"Server IP address: {ip_address}")
    logger.info("Using pre-downloaded yolov5s.pt model")
    logger.info(f"Access the interface at http://{ip_address}:5000")
    logger.info("Other devices on the same network can access this interface using the same URL")
    
    # Run the Flask app, binding to all network interfaces
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
