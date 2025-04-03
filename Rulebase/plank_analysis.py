import cv2
import mediapipe as mp
import numpy as np

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Hàm tính góc
def calculate_angle(p1, p2, p3):
    a = np.array([p1.x, p1.y]) - np.array([p2.x, p2.y])
    b = np.array([p3.x, p3.y]) - np.array([p2.x, p2.y])
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# Hàm tính khoảng cách
def calculate_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# Kiểm tra Plank
def check_plank(landmarks):
    errors = []
    left_shoulder = landmarks[11]
    left_elbow = landmarks[13]
    left_hip = landmarks[23]
    left_heel = landmarks[29]

    back_angle = calculate_angle(left_shoulder, left_hip, left_heel)
    if back_angle > 10:
        errors.append("Hông quá cao")
    elif back_angle < -10:
        errors.append("Hông quá thấp")

    shoulder_elbow_dist = calculate_distance(left_shoulder, left_elbow)
    if shoulder_elbow_dist > 0.05:
        errors.append("Vai không thẳng hàng với khuỷu tay")

    return "Đúng" if not errors else "Sai", errors

# Xử lý video/webcam
def analyze_plank(source):
    cap = cv2.VideoCapture(source)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 3 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                status, errors = check_plank(landmarks)

                cv2.putText(frame, f"Trang thai: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                for i, error in enumerate(errors):
                    cv2.putText(frame, f"Loi {i+1}: {error}", (10, 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Plank Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Chạy
if __name__ == "__main__":
    source = "plank_video.mp4"  # Thay bằng đường dẫn video hoặc 0 cho webcam
    analyze_plank(source)