from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import asyncio
import json
import base64
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

# Hàm xử lý stream từ webcam (gửi cả video và dữ liệu)
async def process_stream(exercise_type, websocket: WebSocket):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        await websocket.send_text(json.dumps({"error": "Could not open webcam"}))
        return

    error_timestamps = {}
    start_time = asyncio.get_event_loop().time()

    while True:
        ret, frame = cap.read()
        if not ret:
            await websocket.send_text(json.dumps({"error": "Failed to read frame"}))
            break

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

        # Mã hóa frame thành base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Tạo payload gồm dữ liệu và video
        result = {
            "status": status,
            "errors": errors,
            "video_frame": frame_base64
        }
        await websocket.send_text(json.dumps(result))

        await asyncio.sleep(0.1)  # Điều chỉnh tốc độ gửi

    cap.release()

# Endpoint kiểm tra API
@app.get("/")
async def root():
    return {"message": "Exercise Analysis API is running. Use WebSocket endpoints: /squat, /plank, /pushup, /lunges"}

# Endpoints WebSocket
@app.websocket("/squat")
async def squat_endpoint(websocket: WebSocket):
    await websocket.accept()
    await process_stream("squat", websocket)

@app.websocket("/plank")
async def plank_endpoint(websocket: WebSocket):
    await websocket.accept()
    await process_stream("plank", websocket)

@app.websocket("/pushup")
async def pushup_endpoint(websocket: WebSocket):
    await websocket.accept()
    await process_stream("pushup", websocket)

@app.websocket("/lunges")
async def lunges_endpoint(websocket: WebSocket):
    await websocket.accept()
    await process_stream("lunges", websocket)

# Chạy API
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2222)