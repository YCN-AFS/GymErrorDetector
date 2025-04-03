import websockets
import asyncio
import json
import cv2
import numpy as np
import base64

async def test_exercise(exercise_type):
    uri = f"ws://localhost:8000/{exercise_type}"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            result = json.loads(message)

            # Giải mã video frame từ base64
            frame_data = base64.b64decode(result["video_frame"])
            np_data = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

            # Hiển thị frame
            cv2.imshow(f"{exercise_type.capitalize()} Analysis", frame)
            print(f"[{exercise_type.upper()}] Status: {result['status']}, Errors: {result['errors']}")

            # Nhấn 'q' để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

# Chạy client cho squat (thay bằng plank, pushup, lunges nếu muốn)
asyncio.run(test_exercise("squat"))