from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import asyncio
import json
import base64
import numpy as np
from squat_analysis import check_squat
from plank_analysis import check_plank
from pushup_analysis import check_pushup
from lunges_analysis import check_lunges

# Khởi tạo FastAPI
app = FastAPI(title="Exercise Analysis API")

# Khởi tạo MediaPipe Pose và Drawing
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Hàm xử lý khung hình từ client
async def process_frame(frame_data, exercise_type, error_timestamps, start_time):
    # Giải mã base64 và chuyển thành hình ảnh
    img_bytes = base64.b64decode(frame_data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if frame is None:
        return None, "Failed to decode frame", []
    
    # Xử lý khung hình với MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        current_time = asyncio.get_event_loop().time() - start_time
        
        # Gọi hàm check tương ứng
        if exercise_type == "squat":
            status, errors = check_squat(landmarks, error_timestamps, current_time)
        elif exercise_type == "plank":
            status, errors = check_plank(landmarks, error_timestamps, current_time)
        elif exercise_type == "pushup":
            status, errors = check_pushup(landmarks, error_timestamps, current_time)
        elif exercise_type == "lunges":
            status, errors = check_lunges(landmarks, error_timestamps, current_time)
        
        # Vẽ landmarks và thông tin lên frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if errors:
            for i, error in enumerate(errors):
                cv2.putText(frame, f"Error {i+1}: {error}", (10, 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No errors detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        status, errors = "No person detected", []
        cv2.putText(frame, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Mã hóa frame đã xử lý thành base64
    _, buffer = cv2.imencode('.jpg', frame)
    processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return processed_frame_base64, status, errors

# Hàm xử lý WebSocket
async def handle_websocket(websocket: WebSocket, exercise_type):
    await websocket.accept()
    
    error_timestamps = {}
    start_time = asyncio.get_event_loop().time()
    
    try:
        while True:
            # Nhận dữ liệu từ client
            data = await websocket.receive_text()
            data_json = json.loads(data)
            
            if "frame" in data_json:
                # Xử lý khung hình
                processed_frame, status, errors = await process_frame(
                    data_json["frame"], exercise_type, error_timestamps, start_time
                )
                
                if processed_frame:
                    # Gửi kết quả về client
                    result = {
                        "status": status,
                        "errors": errors,
                        "video_frame": processed_frame
                    }
                    await websocket.send_text(json.dumps(result))
                else:
                    await websocket.send_text(json.dumps({
                        "status": "Error",
                        "errors": ["Failed to process frame"],
                        "video_frame": ""
                    }))
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("WebSocket connection closed")

# Endpoint kiểm tra API
@app.get("/")
async def root():
    return {"message": "Exercise Analysis API is running. Use WebSocket endpoints: /squat, /plank, /pushup, /lunges"}

# Endpoints WebSocket
@app.websocket("/squat")
async def squat_endpoint(websocket: WebSocket):
    await handle_websocket(websocket, "squat")

@app.websocket("/plank")
async def plank_endpoint(websocket: WebSocket):
    await handle_websocket(websocket, "plank")

@app.websocket("/pushup")
async def pushup_endpoint(websocket: WebSocket):
    await handle_websocket(websocket, "pushup")

@app.websocket("/lunges")
async def lunges_endpoint(websocket: WebSocket):
    await handle_websocket(websocket, "lunges")

# Chạy API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8031)