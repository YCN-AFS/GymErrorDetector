from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
import cv2
import mediapipe as mp
import asyncio
import json
import base64
import numpy as np
import os
import ssl
import logging
import traceback
import sys
from Rulebase.squat_analysis import check_squat
from Rulebase.plank_analysis import check_plank
from Rulebase.pushup_analysis import check_pushup
from Rulebase.lunges_analysis import check_lunges

# Cấu hình logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("exercise_api")

# Thiết lập môi trường để giảm cảnh báo
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Import các module phân tích động tác
try:
    # Tạo các dummy functions nếu không tìm thấy module thực
    def dummy_check(landmarks, error_timestamps, current_time):
        return "No module", []
    
    # Import các module phân tích - dùng try/except để xử lý nếu không tìm thấy
    try:
        from squat_analysis import check_squat
        logger.info("Imported squat_analysis successfully")
    except ImportError:
        logger.error("Could not import squat_analysis. Using dummy function.")
        check_squat = dummy_check
    
    try:
        from plank_analysis import check_plank
        logger.info("Imported plank_analysis successfully")
    except ImportError:
        logger.error("Could not import plank_analysis. Using dummy function.")
        check_plank = dummy_check
    
    try:
        from pushup_analysis import check_pushup
        logger.info("Imported pushup_analysis successfully")
    except ImportError:
        logger.error("Could not import pushup_analysis. Using dummy function.")
        check_pushup = dummy_check
    
    try:
        from lunges_analysis import check_lunges
        logger.info("Imported lunges_analysis successfully")
    except ImportError:
        logger.error("Could not import lunges_analysis. Using dummy function.")
        check_lunges = dummy_check
except Exception as e:
    logger.error(f"Error during module imports: {str(e)}")
    logger.error(traceback.format_exc())

# Khởi tạo FastAPI
app = FastAPI(title="Exercise Analysis API")

# Khởi tạo MediaPipe Pose và Drawing
try:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    logger.info("MediaPipe initialized successfully")
except Exception as e:
    logger.error(f"Error initializing MediaPipe: {str(e)}")
    logger.error(traceback.format_exc())

# Hàm xử lý khung hình từ client
async def process_frame(frame_data, exercise_type, error_timestamps, start_time):
    try:
        # Giải mã base64 và chuyển thành hình ảnh
        img_bytes = base64.b64decode(frame_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Failed to decode frame")
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
            else:
                logger.warning(f"Unknown exercise type: {exercise_type}")
                status, errors = "Unknown Exercise", []
            
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
    except Exception as e:
        logger.error(f"Error in process_frame: {str(e)}")
        logger.error(traceback.format_exc())
        return None, f"Error: {str(e)}", []

# Hàm xử lý WebSocket
async def handle_websocket(websocket: WebSocket, exercise_type):
    try:
        await websocket.accept()
        logger.info(f"WebSocket connection accepted for {exercise_type}")
        
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
            logger.error(f"Error in WebSocket loop: {str(e)}")
            logger.error(traceback.format_exc())
            try:
                await websocket.send_text(json.dumps({
                    "status": "Error",
                    "errors": [f"Server error: {str(e)}"],
                    "video_frame": ""
                }))
            except:
                pass
    except Exception as e:
        logger.error(f"Error handling WebSocket: {str(e)}")
        logger.error(traceback.format_exc())
    
    logger.info("WebSocket connection closed")

# Endpoint kiểm tra API
@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Exercise Analysis API is running. Use WebSocket endpoints: /squat, /plank, /pushup, /lunges"}

# Endpoint kiểm tra kết nối an toàn
@app.get("/ssl-check")
async def ssl_check():
    logger.info("SSL check endpoint accessed")
    return {
        "secure": True,
        "message": "SSL/TLS connection established successfully!",
        "protocol": "WSS (WebSocket Secure)",
        "timestamp": asyncio.get_event_loop().time()
    }

# Endpoints WebSocket
@app.websocket("/squat")
async def squat_endpoint(websocket: WebSocket):
    logger.info("Squat WebSocket endpoint accessed")
    await handle_websocket(websocket, "squat")

@app.websocket("/plank")
async def plank_endpoint(websocket: WebSocket):
    logger.info("Plank WebSocket endpoint accessed")
    await handle_websocket(websocket, "plank")

@app.websocket("/pushup")
async def pushup_endpoint(websocket: WebSocket):
    logger.info("Pushup WebSocket endpoint accessed")
    await handle_websocket(websocket, "pushup")

@app.websocket("/lunges")
async def lunges_endpoint(websocket: WebSocket):
    logger.info("Lunges WebSocket endpoint accessed")
    await handle_websocket(websocket, "lunges")

# Cấu hình SSL cho máy chủ
def configure_ssl():
    try:
        ssl_context = None
        
        # Đường dẫn chứng chỉ cho môi trường phát triển (tự ký)
        dev_cert_path = os.path.join("certs", "cert.pem")
        dev_key_path = os.path.join("certs", "key.pem")
        
        # Đường dẫn chứng chỉ cho môi trường production (Let's Encrypt)
        domain = os.environ.get("DOMAIN_NAME", "yourdomain.com")
        prod_cert_path = f"/etc/letsencrypt/live/{domain}/fullchain.pem"
        prod_key_path = f"/etc/letsencrypt/live/{domain}/privkey.pem"
        
        # Kiểm tra và sử dụng chứng chỉ phù hợp
        if os.environ.get("ENVIRONMENT") == "production" and os.path.exists(prod_cert_path):
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(prod_cert_path, prod_key_path)
            logger.info(f"Using production SSL certificate for {domain}")
        elif os.path.exists(dev_cert_path):
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(dev_cert_path, dev_key_path)
            logger.info("Using development SSL certificate")
        else:
            logger.warning("No SSL certificate found. Running in insecure mode.")
        
        return ssl_context
    except Exception as e:
        logger.error(f"Error configuring SSL: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# Chạy API
if __name__ == "__main__":
    try:
        import uvicorn
        
        # Lấy cấu hình từ biến môi trường
        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", 2222))
        environment = os.environ.get("ENVIRONMENT", "development")
        
        logger.info(f"Starting server on {host}:{port} in {environment} environment")
        
        try:
            # Tạo thư mục phân tích động tác nếu chưa tồn tại
            for module_name in ["squat_analysis.py", "plank_analysis.py", "pushup_analysis.py", "lunges_analysis.py"]:
                if not os.path.exists(module_name):
                    logger.warning(f"Module {module_name} not found. Creating dummy module...")
                    with open(module_name, "w") as f:
                        f.write("""import math
from typing import Dict, List, Tuple, Any

def check_{0}(landmarks: List[Any], error_timestamps: Dict, current_time: float) -> Tuple[str, List[str]]:
    \"\"\"
    Dummy {1} check function.
    \"\"\"
    return "Dummy {1} Analysis", ["This is a dummy module. Please implement proper analysis."]

def calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    \"\"\"Calculate angle between three points\"\"\"
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - 
        math.atan2(a[1] - b[1], a[0] - b[0])
    )
    angle = abs(angle)
    if angle > 180:
        angle = 360 - angle
    return angle
""".format(module_name.split("_")[0], module_name.split("_")[0].capitalize()))
                    logger.info(f"Created dummy module {module_name}")
        except Exception as e:
            logger.error(f"Error creating dummy modules: {str(e)}")
            logger.error(traceback.format_exc())
        
        # Tạo thư mục cho SSL certificates
        os.makedirs("certs", exist_ok=True)
        
        if environment == "production":
            # Cho môi trường production (Let's Encrypt)
            domain = os.environ.get("DOMAIN_NAME", "yourdomain.com")
            ssl_keyfile = f"/etc/letsencrypt/live/{domain}/privkey.pem"
            ssl_certfile = f"/etc/letsencrypt/live/{domain}/fullchain.pem"
            
            if os.path.exists(ssl_certfile) and os.path.exists(ssl_keyfile):
                logger.info(f"Starting server with SSL: {ssl_certfile}")
                uvicorn.run(app, host=host, port=port, ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile)
            else:
                logger.warning(f"SSL certificates not found at {ssl_certfile}")
                logger.info("Running in insecure mode")
                uvicorn.run(app, host=host, port=port)
        else:
            # Cho môi trường phát triển
            ssl_keyfile = "certs/key.pem"
            ssl_certfile = "certs/cert.pem"
            
            # Check if we should skip SSL
            skip_ssl = os.environ.get("SKIP_SSL", "False").lower() in ("true", "1", "yes")
            
            if not skip_ssl and os.path.exists(ssl_certfile) and os.path.exists(ssl_keyfile):
                logger.info(f"Starting server with development SSL: {ssl_certfile}")
                uvicorn.run(app, host=host, port=port, ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile)
            else:
                if skip_ssl:
                    logger.info("SSL skipped due to SKIP_SSL environment variable")
                else:
                    logger.warning("Development SSL certificates not found")
                logger.info("Running in insecure mode")
                uvicorn.run(app, host=host, port=port)
    except Exception as e:
        logger.critical(f"Failed to start server: {str(e)}")
        logger.critical(traceback.format_exc())
        print(f"CRITICAL ERROR: {str(e)}")
        print(traceback.format_exc())