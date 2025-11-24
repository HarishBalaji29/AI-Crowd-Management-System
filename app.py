# app.py
# Main application for the AI-Based CCTV Network for Crowd Management with Gemini Integration

import cv2
import numpy as np
import os
from flask import Flask, render_template, Response, jsonify, request
import time
import json
import requests
# Optional dotenv support: try to import python-dotenv, otherwise provide a noop for load_dotenv
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*args, **kwargs):
        # dotenv is optional; environment variables can be set in the environment instead.
        return None

# Initialize the Flask application
app = Flask(__name__)

# load environment variables from .env (if present)
load_dotenv()

# --- Configuration ---
# Load the pre-trained MobileNet SSD model for person detection
PROTOTXT = "deploy.prototxt"
MODEL = "Caffe model file.caffemodel"

# Validate model files exist to give a clearer error if they are missing
if not os.path.exists(PROTOTXT):
    raise FileNotFoundError(f"Prototxt file not found: {PROTOTXT}")
if not os.path.exists(MODEL):
    raise FileNotFoundError(f"Model file not found: {MODEL}")

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# Confidence threshold for detections
CONFIDENCE_THRESHOLD = 0.4

# Define the classes the model can detect. We are only interested in 'person'.
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# --- Global State for Status ---
# This dictionary will hold the latest analysis results to be accessed by different routes.
# This is a simple approach for this example. In a production app, you might use a more robust state management solution.
app_status = {
    "person_count": 0,
    "density_level": "Low",
    "last_updated": time.time()
}

# --- Crowd Density Logic ---
def analyze_frame(frame):
    """
    Analyzes a single video frame to detect people and estimate crowd density.
    Updates the global app_status.

    Args:
        frame: The input video frame (a NumPy array).

    Returns:
        The processed frame with bounding boxes and density info.
    """
    global app_status
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()
    person_count = 0

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > CONFIDENCE_THRESHOLD:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "person":
                person_count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Determine the crowd density level
    if person_count < 5:
        density_level = "Low"
        color = (0, 255, 0)
    elif person_count < 10:
        density_level = "Medium"
        color = (0, 255, 255)
    else:
        density_level = "High"
        color = (0, 0, 255)

    # Update global status
    app_status['person_count'] = person_count
    app_status['density_level'] = density_level
    app_status['last_updated'] = time.time()

    # Display info on the frame
    cv2.putText(frame, f"Person Count: {person_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, f"Density: {density_level}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame

def generate_frames():
    """Generator function for video streaming."""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            processed_frame = analyze_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    camera.release()

# --- Flask Routes ---
@app.route('/')
def index():
    """Render the main web page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    """Endpoint to get the current crowd status."""
    return jsonify(app_status)

@app.route('/generate_protocol', methods=['POST'])
def generate_protocol():
    """
    Endpoint to generate a safety protocol using the Gemini API.
    """
    data = request.get_json()
    person_count = data.get('person_count', 0)

    prompt = f"""
    You are a security and safety expert. A CCTV system has detected a high-density crowd of approximately {person_count} people in a public area.
    Generate a concise, clear, and actionable safety protocol for on-ground security personnel.
    The protocol should be formatted in Markdown and include the following sections:
    1.  **Immediate Actions:** 3-4 critical first steps.
    2.  **Communication Protocol:** Who to contact and what to report.
    3.  **Crowd Management Techniques:** 2-3 specific techniques to de-escalate the situation.
    4.  **Emergency Preparedness:** Key reminders for potential evacuation or medical needs.
    """

    # Read API key from environment variable `GEN_API_KEY` and optional mock mode `MOCK_GEMINI`.
    api_key = os.getenv('GEN_API_KEY')
    # sanitize: remove accidental surrounding quotes and whitespace
    if api_key:
        api_key = api_key.strip()
        if (api_key.startswith('"') and api_key.endswith('"')) or (api_key.startswith("'") and api_key.endswith("'")):
            api_key = api_key[1:-1].strip()
    mock_mode = os.getenv('MOCK_GEMINI', 'false').lower() in ('1', 'true', 'yes')
     # Allow forcing mock output via the request body: {"mock": true}
    force_mock = bool(data.get('mock', False))
    # quick placeholder detection to avoid sending invalid keys
    if api_key and ('your_real' in api_key.lower() or 'replace' in api_key.lower() or len(api_key) < 10):
        # treat as missing/invalid key for guidance
        api_key = None

    # A small canned protocol used when mock mode is enabled for local development/testing.
    canned_protocol = (
        "### Immediate Actions:\n"
        "1. Secure the perimeter and assign staff to form clear ingress/egress paths.\n"
        "2. Call local law enforcement if crowd exceeds safe thresholds.\n"
        "3. Begin staged, calm announcements to guide people to less crowded areas.\n\n"
        "### Communication Protocol:\n"
        "- Notify on-site supervisor and control room immediately.\n"
        "- Report estimated person count, location, and any injuries.\n\n"
        "### Crowd Management Techniques:\n"
        "- Use clear, calm public address instructions to direct flows.\n"
        "- Open additional exits and direct flow to alternate routes.\n\n"
        "### Emergency Preparedness:\n"
        "- Prepare medical teams and designate evacuation zones.\n"
        "- Keep clear channels for emergency responders.\n"
    )

    # If no key, allow mock modes (env, local file fallback already tried, or request mock)
    if not api_key:
        # Allow mock via env, request, or automatic fallback in debug/local development.
        if mock_mode or force_mock or app.debug:
            return jsonify({'protocol': canned_protocol})
        return jsonify({
            'error': "Gemini API key not configured. Set GEN_API_KEY in a .env file or environment, enable MOCK_GEMINI, or send {\"mock\": true}."
        }), 400

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }]
    }
    
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=30)
        try:
            response.raise_for_status() # Raise an exception for bad status codes
        except requests.exceptions.HTTPError:
            status = response.status_code
            # Special guidance for 403 Forbidden
            if status == 403:
                if mock_mode or app.debug:
                    return jsonify({'protocol': canned_protocol})
                # Provide actionable guidance without leaking details
                return jsonify({
                    'error': 'API request returned 403 Forbidden.',
                    'help': [
                        'Confirm GEN_API_KEY is correct (no surrounding quotes) and not a placeholder.',
                        'Ensure the Generative Language API (Generative) is enabled in your Google Cloud project.',
                        'Verify billing is enabled for the project and the key is unrestricted or allows this request origin.',
                        'Check key restrictions (HTTP referrer / IP / service restrictions) and remove them for testing.'
                    ]
                }), 403
            return jsonify({'error': f'API request failed with status {status}: {response.text}'}), status

        result = response.json()
        
        if (result.get('candidates') and result['candidates'][0].get('content') and 
            result['candidates'][0]['content'].get('parts') and result['candidates'][0]['content']['parts'][0].get('text')):
            
            generated_text = result['candidates'][0]['content']['parts'][0]['text']
            return jsonify({'protocol': generated_text})
        else:
            # Handle cases where the response structure is unexpected
            error_message = "Error: Could not parse the response from the Gemini API."
            if result.get('promptFeedback'):
                error_message += f" Reason: {result['promptFeedback'].get('blockReason')}"
            return jsonify({'error': error_message}), 500

    except requests.exceptions.RequestException as e:
        return jsonify({'error': f"API request failed: {e}"}), 500


# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True)
