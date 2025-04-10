from fastapi import FastAPI, WebSocket, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import mediapipe as mp
import asyncio
import json
import base64
import numpy as np
import os
import uuid
from .squat_analysis import check_squat
from .plank_analysis import check_plank
from .pushup_analysis import check_pushup
from .lunges_analysis import check_lunges

# Khởi tạo FastAPI
app = FastAPI(title="Exercise Analysis API")

# Thêm CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tạo thư mục lưu trữ video nếu chưa tồn tại
VIDEO_UPLOAD_DIR = "uploaded_videos"
VIDEO_PROCESSED_DIR = "processed_videos"
os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)
os.makedirs(VIDEO_PROCESSED_DIR, exist_ok=True)

# Khởi tạo MediaPipe Pose và Drawing
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def process_video(input_path, exercise_type):
    """
    Xử lý video và thêm phân tích chuyển động
    """
    # Mở video đầu vào
    cap = cv2.VideoCapture(input_path)
    
    # Tạo video writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Tạo tên file output duy nhất
    output_filename = f"{uuid.uuid4()}_processed.mp4"
    output_path = os.path.join(VIDEO_PROCESSED_DIR, output_filename)
    
    # Tạo video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Khởi tạo các biến theo dõi
    error_timestamps = {}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rep_count = 0
    start_time = 0
    
    # Xử lý từng khung hình
    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Xử lý khung hình với MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Gọi hàm check tương ứng
            if exercise_type == "squat":
                status, errors, current_rep_count = check_squat(landmarks, error_timestamps, frame_num/fps)
                rep_count = current_rep_count
            elif exercise_type == "plank":
                status, errors = check_plank(landmarks, error_timestamps, frame_num/fps)
            elif exercise_type == "pushup":
                status, errors = check_pushup(landmarks, error_timestamps, frame_num/fps)
            elif exercise_type == "lunges":
                status, errors = check_lunges(landmarks, error_timestamps, frame_num/fps)
            
            # Vẽ landmarks và thông tin lên frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # Hiển thị trạng thái
            cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Reps: {rep_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Hiển thị lỗi
            if errors:
                for i, error in enumerate(errors):
                    cv2.putText(frame, f"Error {i+1}: {error}", (10, 90 + i*30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Ghi khung hình đã xử lý
        out.write(frame)
    
    # Giải phóng tài nguyên
    cap.release()
    out.release()
    
    return output_path, rep_count

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...), exercise_type: str = "squat"):
    """
    Endpoint tải lên video và xử lý
    """
    # Kiểm tra loại file
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file.")
    
    # Tạo tên file duy nhất
    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(VIDEO_UPLOAD_DIR, filename)
    
    # Lưu file tải lên
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # Xử lý video
        processed_video_path, rep_count = process_video(file_path, exercise_type)
        
        # Trả về thông tin video đã xử lý
        return {
            "message": "Video processed successfully",
            "processed_video": processed_video_path.split('/')[-1],
            "rep_count": rep_count
        }
    except Exception as e:
        # Xóa file gốc nếu xảy ra lỗi
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.get("/download-video/{filename}")
async def download_video(filename: str):
    """
    Endpoint tải xuống video đã xử lý
    """
    file_path = os.path.join(VIDEO_PROCESSED_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path, 
        media_type='video/mp4', 
        filename=filename
    )

# Endpoint kiểm tra API
@app.get("/")
async def root():
    return {
        "message": "Exercise Analysis API is running",
        "endpoints": {
            "upload_video": "/upload-video/",
            "download_video": "/download-video/{filename}",
            "websocket_endpoints": ["/squat", "/plank", "/pushup", "/lunges"]
        }
    }

# Hàm xử lý khung hình từ client
async def process_frame(frame_data, exercise_type, error_timestamps, start_time):
    # Giải mã base64 và chuyển thành hình ảnh
    img_bytes = base64.b64decode(frame_data)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if frame is None:
        return None, "Failed to decode frame", [], 0
    
    # Xử lý khung hình với MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        current_time = asyncio.get_event_loop().time() - start_time
        
        # Gọi hàm check tương ứng
        if exercise_type == "squat":
            status, errors, rep_count = check_squat(landmarks, error_timestamps, current_time)
        elif exercise_type == "plank":
            status, errors = check_plank(landmarks, error_timestamps, current_time)
            rep_count = 0  # Plank doesn't have rep count
        elif exercise_type == "pushup":
            status, errors = check_pushup(landmarks, error_timestamps, current_time)
            rep_count = 0  # Pushup might need separate rep counting logic
        elif exercise_type == "lunges":
            status, errors = check_lunges(landmarks, error_timestamps, current_time)
            rep_count = 0  # Lunges might need separate rep counting logic
        
        # Vẽ landmarks và thông tin lên frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Reps: {rep_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if errors:
            for i, error in enumerate(errors):
                cv2.putText(frame, f"Error {i+1}: {error}", (10, 90 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No errors detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        status, errors, rep_count = "No person detected", [], 0
        cv2.putText(frame, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Mã hóa frame đã xử lý thành base64
    _, buffer = cv2.imencode('.jpg', frame)
    processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return processed_frame_base64, status, errors, rep_count

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
                processed_frame, status, errors, rep_count = await process_frame(
                    data_json["frame"], exercise_type, error_timestamps, start_time
                )
                
                if processed_frame:
                    # Gửi kết quả về client
                    result = {
                        "status": status,
                        "errors": errors,
                        "rep_count": rep_count,
                        "video_frame": processed_frame
                    }
                    await websocket.send_text(json.dumps(result))
                else:
                    await websocket.send_text(json.dumps({
                        "status": "Error",
                        "errors": ["Failed to process frame"],
                        "rep_count": 0,
                        "video_frame": ""
                    }))
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("WebSocket connection closed")

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
    uvicorn.run(app, host="0.0.0.0", port=2222, 
                # Add additional configuration for broader network access
                workers=4,  # Multiple workers for better performance
                proxy_headers=True,  # Handle proxy headers
                forwarded_allow_ips="*"  # Allow all IPs through proxy
    )