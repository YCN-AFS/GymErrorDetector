import cv2
import mediapipe as mp
import numpy as np

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils  # Để vẽ keypoints (tùy chọn)

# Hàm tính góc giữa 3 điểm
def calculate_angle(p1, p2, p3):
    a = np.array([p1.x, p1.y]) - np.array([p2.x, p2.y])
    b = np.array([p3.x, p3.y]) - np.array([p2.x, p2.y])
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# Hàm tính khoảng cách giữa 2 điểm
def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# Hàm kiểm tra Squat
def check_squat(landmarks):
    errors = []
    # Keypoints
    left_shoulder = landmarks[11]
    left_hip = landmarks[23]
    left_knee = landmarks[25]
    left_ankle = landmarks[27]
    left_toe = landmarks[31]

    # Góc lưng
    back_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    if back_angle < 160:
        errors.append("Lưng cong")

    # Đầu gối vượt mũi chân
    if left_knee.x > left_toe.x + 0.1:
        errors.append("Đầu gối vượt quá mũi chân")

    # Góc đầu gối
    knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    if knee_angle > 90:
        errors.append("Hông không xuống đủ sâu")

    return "Đúng" if not errors else "Sai", errors

# Hàm kiểm tra Plank
def check_plank(landmarks):
    errors = []
    # Keypoints
    left_shoulder = landmarks[11]
    left_elbow = landmarks[13]
    left_hip = landmarks[23]
    left_heel = landmarks[29]

    # Góc lưng
    back_angle = calculate_angle(left_shoulder, left_hip, left_heel)
    if back_angle > 10:
        errors.append("Hông quá cao")
    elif back_angle < -10:
        errors.append("Hông quá thấp")

    # Vai thẳng khuỷu tay
    shoulder_elbow_dist = calculate_distance(left_shoulder, left_elbow)
    if shoulder_elbow_dist > 0.05:  # Khoảng cách dọc
        errors.append("Vai không thẳng hàng với khuỷu tay")

    return "Đúng" if not errors else "Sai", errors

# Hàm kiểm tra Push-up
def check_pushup(landmarks):
    errors = []
    # Keypoints
    left_shoulder = landmarks[11]
    left_elbow = landmarks[13]
    left_wrist = landmarks[15]
    left_hip = landmarks[23]
    left_heel = landmarks[29]

    # Góc lưng
    back_angle = calculate_angle(left_shoulder, left_hip, left_heel)
    if back_angle > 15:
        errors.append("Hông quá cao")
    elif back_angle < -15:
        errors.append("Hông quá thấp")

    # Góc khuỷu tay (ở điểm thấp nhất)
    elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    if elbow_angle > 90:
        errors.append("Khuỷu tay không gập đủ")

    return "Đúng" if not errors else "Sai", errors

# Hàm chính để xử lý video
def analyze_exercise(video_path, exercise_type="squat"):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển frame sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Chọn hàm kiểm tra theo loại động tác
            if exercise_type == "squat":
                status, errors = check_squat(landmarks)
            elif exercise_type == "plank":
                status, errors = check_plank(landmarks)
            elif exercise_type == "pushup":
                status, errors = check_pushup(landmarks)
            else:
                status, errors = "Không xác định", ["Loại động tác không hỗ trợ"]

            # Hiển thị kết quả trên frame
            cv2.putText(frame, f"Trang thai: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            for i, error in enumerate(errors):
                cv2.putText(frame, f"Loi {i+1}: {error}", (10, 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Vẽ keypoints (tùy chọn)
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Exercise Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Chạy thử
if __name__ == "__main__":
    # Thay đường dẫn video của mày vào đây
    video_path = "squat_video.mp4"  # Ví dụ
    analyze_exercise(video_path, exercise_type="squat")  # Chọn "squat", "plank", hoặc "pushup"