import cv2
import mediapipe as mp

# Khởi tạo pose và drawing utils
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
from mediapipe.framework.formats import landmark_pb2

# Danh sách ID các điểm trên mặt trong pose
FACE_LANDMARK_IDS = list(range(0, 11)) + list(range(15, 23))

# Định nghĩa màu và độ dày riêng cho đùi
CUSTOM_CONNECTIONS_STYLE = {
    # Bỏ comment 2 dòng dưới để đổi màu đùi trái
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE): mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4),
    
    # Bỏ comment 2 dòng dưới để đổi màu đùi phải
    # (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE): mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4),
}

# Đường dẫn ảnh
input_path = r"C:\Users\fox\Downloads\z6475208591853_0f5537901cb75556fddb5921c600b0ce.jpg"
output_path = 'output_pose.jpg'

# Đọc ảnh và chuyển sang RGB
image = cv2.imread(input_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Xử lý pose
with mp_pose.Pose(static_image_mode=True) as pose:
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # Tạo bản sao ảnh
        annotated_image = image.copy()

        # Loại bỏ các điểm trên mặt bằng cách thay landmark đó bằng điểm "vô hình"
        new_landmarks = landmark_pb2.NormalizedLandmarkList()
        for idx, lm in enumerate(results.pose_landmarks.landmark):
            if idx in FACE_LANDMARK_IDS:
                invisible = landmark_pb2.NormalizedLandmark(x=0, y=0, z=0, visibility=0)
                new_landmarks.landmark.append(invisible)
            else:
                new_landmarks.landmark.append(lm)

        # Vẽ toàn bộ các connections
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start_lm = new_landmarks.landmark[start_idx]
            end_lm = new_landmarks.landmark[end_idx]

            # Bỏ qua nếu điểm không hiển thị
            if start_lm.visibility < 0.5 or end_lm.visibility < 0.5:
                continue

            # Tính tọa độ điểm
            h, w, _ = image.shape
            start_point = (int(start_lm.x * w), int(start_lm.y * h))
            end_point = (int(end_lm.x * w), int(end_lm.y * h))

            # Kiểm tra xem có dùng style riêng không
            if (start_idx, end_idx) in CUSTOM_CONNECTIONS_STYLE:
                style = CUSTOM_CONNECTIONS_STYLE[(start_idx, end_idx)]
                cv2.line(annotated_image, start_point, end_point, style.color, style.thickness)
            elif (end_idx, start_idx) in CUSTOM_CONNECTIONS_STYLE:
                style = CUSTOM_CONNECTIONS_STYLE[(end_idx, start_idx)]
                cv2.line(annotated_image, start_point, end_point, style.color, style.thickness)
            else:
                # Dùng style mặc định
                cv2.line(annotated_image, start_point, end_point, (255, 255, 255), 2)

        # Lưu kết quả
        cv2.imwrite(output_path, annotated_image)
        print(f"Đã lưu ảnh có pose (không vẽ mặt) tại: {output_path}")
    else:
        print("Không phát hiện pose trong ảnh.")
