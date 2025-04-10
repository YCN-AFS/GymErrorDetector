import websockets
import asyncio
import base64
import cv2
import numpy as np
import json
import ssl

async def analyze_exercise(exercise_type):
    # Endpoint API công khai
    uri = f"wss://cf.s4h.edu.vn/{exercise_type}"

    # Cấu hình SSL linh hoạt
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    try:
        # Kết nối WebSocket
        async with websockets.connect(uri, ssl=ssl_context) as websocket:
            # Mở camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Không thể mở camera.")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Không thể đọc khung hình.")
                    break

                # Nén và chuyển đổi frame
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                try:
                    # Gửi frame
                    await websocket.send(json.dumps({"frame": frame_base64}))
                    
                    # Nhận kết quả
                    response = await websocket.recv()
                    result = json.loads(response)

                    # Hiển thị kết quả
                    print(f"Status: {result.get('status', 'Unknown')}")
                    print(f"Errors: {result.get('errors', [])}")

                    # Hiển thị video frame đã xử lý
                    if result.get('video_frame'):
                        processed_frame = base64.b64decode(result['video_frame'])
                        processed_frame_np = cv2.imdecode(np.frombuffer(processed_frame, np.uint8), cv2.IMREAD_COLOR)
                        cv2.imshow('Exercise Analysis', processed_frame_np)

                except Exception as e:
                    print(f"Lỗi xử lý frame: {e}")
                
                # Thoát khi nhấn 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Giải phóng tài nguyên
            cap.release()
            cv2.destroyAllWindows()

    except websockets.exceptions.WebSocketException as e:
        print(f"Lỗi WebSocket: {e}")
    except Exception as e:
        print(f"Lỗi không xác định: {e}")

# Hàm chính để chạy phân tích
async def main():
    # Danh sách các bài tập để người dùng lựa chọn
    exercises = ['squat', 'plank', 'pushup', 'lunges']
    
    print("Các bài tập khả dụng:")
    for i, exercise in enumerate(exercises, 1):
        print(f"{i}. {exercise.capitalize()}")
    
    # Lựa chọn bài tập
    choice = input("Chọn số thứ tự bài tập: ")
    try:
        exercise_type = exercises[int(choice) - 1]
        await analyze_exercise(exercise_type)
    except (ValueError, IndexError):
        print("Lựa chọn không hợp lệ.")

# Chạy chương trình
if __name__ == "__main__":
    asyncio.run(main())