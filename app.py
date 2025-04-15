# app.py
from flask import Flask, render_template, request, jsonify, Response
import os
import cv2
import mediapipe as mp
import base64
import numpy as np
import json
import time
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Clean up uploads directory on startup
directory_path = 'uploads'
for filename in os.listdir(directory_path):
    file_path = os.path.join(directory_path, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)  # Delete file or link
        elif os.path.isdir(file_path):
            os.rmdir(file_path)  # Delete directory if empty
    except Exception as e:
        # print(f'Cannot delete {file_path}. Error: {e}')
        pass

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Dictionary to store exercise sessions
exercise_sessions = {}

# Exercise analysis functions
def check_squat(landmarks, error_timestamps, current_time):
    """Analyze squat form using landmarks"""
    # Get key landmarks
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    
    # Calculate angles
    hip_angle = calculate_angle(shoulder, hip, knee)
    knee_angle = calculate_angle(hip, knee, ankle)
    
    errors = []
    status = "Good form"
    
    # Check if in squat position (knees bent)
    is_squatting = knee_angle < 120
    
    # Check for common errors
    if knee_angle < 70:
        error = "Knees bent too much"
        if "knee_bend" not in error_timestamps:
            error_timestamps["knee_bend"] = current_time
        errors.append(error)
        status = "Fix your form"
    
    if hip_angle > 120 and is_squatting:
        error = "Back not straight"
        if "back_straight" not in error_timestamps:
            error_timestamps["back_straight"] = current_time
        errors.append(error)
        status = "Fix your form"
    
    # Simple rep counter (basic implementation)
    rep_count = 0
    if "session_data" in error_timestamps:
        session_data = error_timestamps["session_data"]
        if is_squatting and not session_data.get("was_squatting", False):
            session_data["was_squatting"] = True
        elif not is_squatting and session_data.get("was_squatting", False):
            session_data["was_squatting"] = False
            session_data["rep_count"] = session_data.get("rep_count", 0) + 1
        rep_count = session_data.get("rep_count", 0)
    else:
        error_timestamps["session_data"] = {"was_squatting": is_squatting, "rep_count": 0}
    
    return status, errors, rep_count

def check_plank(landmarks, error_timestamps, current_time):
    """Analyze plank form using landmarks"""
    # Get key landmarks
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    
    # Calculate body alignment
    alignment_angle = calculate_angle(shoulder, hip, ankle)
    
    errors = []
    status = "Good form"
    
    # Check for common errors
    if not (160 <= alignment_angle <= 180):
        error = "Body not aligned straight"
        if "body_alignment" not in error_timestamps:
            error_timestamps["body_alignment"] = current_time
        errors.append(error)
        status = "Fix your form"
    
    if hip.y > shoulder.y + 0.1:
        error = "Hips too high"
        if "hips_high" not in error_timestamps:
            error_timestamps["hips_high"] = current_time
        errors.append(error)
        status = "Fix your form"
    
    if hip.y < shoulder.y - 0.1:
        error = "Hips too low"
        if "hips_low" not in error_timestamps:
            error_timestamps["hips_low"] = current_time
        errors.append(error)
        status = "Fix your form"
    
    return status, errors

def check_pushup(landmarks, error_timestamps, current_time):
    """Analyze pushup form using landmarks"""
    # Get key landmarks
    shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    
    # Calculate angles
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    body_angle = calculate_angle(shoulder, hip, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    
    errors = []
    status = "Good form"
    
    # Check if in pushup down position
    is_down = elbow_angle < 90
    
    # Check for common errors
    if not (160 <= body_angle <= 180):
        error = "Body not straight"
        if "body_straight" not in error_timestamps:
            error_timestamps["body_straight"] = current_time
        errors.append(error)
        status = "Fix your form"
    
    if hip.y < shoulder.y - 0.1:
        error = "Hips too low"
        if "hips_low" not in error_timestamps:
            error_timestamps["hips_low"] = current_time
        errors.append(error)
        status = "Fix your form"
    
    # Simple rep counter logic
    if "session_data" in error_timestamps:
        session_data = error_timestamps["session_data"]
        if is_down and not session_data.get("was_down", False):
            session_data["was_down"] = True
        elif not is_down and session_data.get("was_down", False):
            session_data["was_down"] = False
            session_data["rep_count"] = session_data.get("rep_count", 0) + 1
    else:
        error_timestamps["session_data"] = {"was_down": is_down, "rep_count": 0}
    
    return status, errors

def check_lunges(landmarks, error_timestamps, current_time):
    """Analyze lunges form using landmarks"""
    # Get key landmarks
    hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    
    # Calculate angles
    knee_angle = calculate_angle(hip, knee, ankle)
    
    errors = []
    status = "Good form"
    
    # Check for common errors
    if knee_angle < 70:
        error = "Front knee bent too much"
        if "knee_bend" not in error_timestamps:
            error_timestamps["knee_bend"] = current_time
        errors.append(error)
        status = "Fix your form"
    
    # Track if person is in lunge position
    is_lunging = knee_angle < 130
    
    # Simple rep counter logic
    if "session_data" in error_timestamps:
        session_data = error_timestamps["session_data"]
        if is_lunging and not session_data.get("was_lunging", False):
            session_data["was_lunging"] = True
        elif not is_lunging and session_data.get("was_lunging", False):
            session_data["was_lunging"] = False
            session_data["rep_count"] = session_data.get("rep_count", 0) + 1
    else:
        error_timestamps["session_data"] = {"was_lunging": is_lunging, "rep_count": 0}
    
    return status, errors

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def process_frame(frame_data, exercise_type, session_id):
    """Process a video frame and analyze exercise form"""
    # Get or create session data
    if session_id not in exercise_sessions:
        exercise_sessions[session_id] = {
            'error_timestamps': {'session_data': {'rep_count': 0}},
            'start_time': time.time()
        }
    
    session = exercise_sessions[session_id]
    error_timestamps = session['error_timestamps']
    start_time = session['start_time']
    current_time = time.time() - start_time
    
    # Decode base64 to image
    try:
        img_data = base64.b64decode(frame_data.split(',')[1] if ',' in frame_data else frame_data)
        img_array = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return None, "Failed to decode frame", [], 0
    
    if frame is None:
        return None, "Invalid frame", [], 0
    
    # Process with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if not results.pose_landmarks:
        return encode_frame(frame), "No person detected", [], 0
    
    landmarks = results.pose_landmarks.landmark
    
    # Analyze exercise based on type
    if exercise_type == "squat":
        status, errors, rep_count = check_squat(landmarks, error_timestamps, current_time)
    elif exercise_type == "plank":
        status, errors = check_plank(landmarks, error_timestamps, current_time)
        rep_count = 0  # Plank doesn't count reps
    elif exercise_type == "pushup":
        status, errors = check_pushup(landmarks, error_timestamps, current_time)
        rep_count = error_timestamps.get("session_data", {}).get("rep_count", 0)
    elif exercise_type == "lunges":
        status, errors = check_lunges(landmarks, error_timestamps, current_time)
        rep_count = error_timestamps.get("session_data", {}).get("rep_count", 0)
    else:
        return encode_frame(frame), "Unknown exercise type", [], 0
    
    # Draw pose landmarks
    mp_drawing.draw_landmarks(
        frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2, circle_radius=1)
    )
    
    # Add info to frame
    cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Reps: {rep_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if errors:
        for i, error in enumerate(errors):
            cv2.putText(frame, f"Error: {error}", (10, 90 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "No errors detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return encode_frame(frame), status, errors, rep_count

def encode_frame(frame):
    """Encode frame as base64 string"""
    _, buffer = cv2.imencode('.jpg', frame)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    frame_data = data.get('frame')
    exercise_type = data.get('exercise_type', 'squat')
    session_id = data.get('session_id', 'default')
    
    if not frame_data:
        return jsonify({'error': 'No frame data provided'}), 400
    
    processed_frame, status, errors, rep_count = process_frame(frame_data, exercise_type, session_id)
    
    if processed_frame is None:
        return jsonify({'error': 'Failed to process frame'}), 400
    
    return jsonify({
        'status': status,
        'errors': errors,
        'rep_count': rep_count,
        'processed_frame': processed_frame
    })

@app.route('/start_session', methods=['POST'])
def start_session():
    data = request.json
    exercise_type = data.get('exercise_type', 'squat')
    session_id = f"{exercise_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    exercise_sessions[session_id] = {
        'error_timestamps': {'session_data': {'rep_count': 0}},
        'start_time': time.time(),
        'exercise_type': exercise_type
    }
    
    return jsonify({
        'session_id': session_id,
        'message': f'Started new {exercise_type} session'
    })

@app.route('/end_session', methods=['POST'])
def end_session():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id in exercise_sessions:
        session_data = exercise_sessions.pop(session_id)
        rep_count = session_data['error_timestamps'].get('session_data', {}).get('rep_count', 0)
        return jsonify({
            'message': 'Session ended',
            'session_id': session_id,
            'total_reps': rep_count,
            'duration': time.time() - session_data['start_time']
        })
    
    return jsonify({'error': 'Session not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8031, debug=True)