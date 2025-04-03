import cv2
import mediapipe as mp
import numpy as np
import time

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Hàm tính góc
def calculate_angle(p1, p2, p3):
    a = np.array([p1.x, p1.y]) - np.array([p2.x, p2.y])
    b = np.array([p3.x, p3.y]) - np.array([p2.x, p2.y])
    cosine_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

# Hàm tính góc nghiêng so với trục X (mặt đất)
def calculate_tilt_angle(p1, p2):
    vector = np.array([p2[0] - p1[0], p2[1] - p1[1]])  # Hông → Vai
    x_axis = np.array([1, 0])  # Trục X hướng phải
    cosine_angle = np.dot(vector, x_axis) / (np.linalg.norm(vector) * np.linalg.norm(x_axis))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    # Điều chỉnh góc nếu vector ngược chiều trục X
    if p2[0] < p1[0]:  # Vai bên trái hông
        return 180 - angle
    return angle

# Kiểm tra Plank
def check_plank(landmarks, error_timestamps, current_time):
    potential_errors = {}
    confirmed_errors = []
    left_shoulder = landmarks[11]
    left_elbow = landmarks[13]
    left_wrist = landmarks[15]
    left_hip = landmarks[23]
    left_knee = landmarks[25]
    left_ankle = landmarks[27]
    right_shoulder = landmarks[12]
    right_elbow = landmarks[14]
    right_wrist = landmarks[16]
    right_hip = landmarks[24]
    right_knee = landmarks[26]
    right_ankle = landmarks[28]

    try:
        # Tính góc lưng (vai-hông-đầu gối)
        left_back_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_back_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        back_angle = (left_back_angle + right_back_angle) / 2

        # Tính góc đầu gối
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        knee_angle = (left_knee_angle + right_knee_angle) / 2

        # Tính góc khuỷu tay
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        elbow_angle = (left_elbow_angle + right_elbow_angle) / 2

        # Tính độ nghiêng cơ thể so với trục X
        mid_shoulder = np.array([(left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2])
        mid_hip = np.array([(left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2])
        tilt_angle = calculate_tilt_angle(mid_hip, mid_shoulder)

        print(f"Back angle: {back_angle:.2f}°, Knee angle: {knee_angle:.2f}°, Elbow angle: {elbow_angle:.2f}°, "
              f"Tilt angle: {tilt_angle:.2f}°, Shoulder Y: {mid_shoulder[1]:.2f}, Hip Y: {mid_hip[1]:.2f}")

        # Kiểm tra tư thế không hợp lệ
        if tilt_angle > 20:
            potential_errors["Body tilted"] = f"Tilt angle: {tilt_angle:.2f}° > 20°"
        if mid_shoulder[1] > mid_hip[1] + 0.05:
            potential_errors["Not in plank position"] = f"Shoulder Y: {mid_shoulder[1]:.2f} > Hip Y: {mid_hip[1]:.2f}"

        # Kiểm tra lỗi plank
        if back_angle < 165:  # Nới lỏng từ 160° lên 165°
            potential_errors["Back too low"] = f"Back angle: {back_angle:.2f}° < 165°"
        elif back_angle > 200:
            potential_errors["Hips too high"] = f"Back angle: {back_angle:.2f}° > 200°"

        if knee_angle < 150 or left_knee.y > 0.9:
            potential_errors["Knees too low"] = f"Knee angle: {knee_angle:.2f}° < 150° or Knee Y: {left_knee.y:.2f} > 0.9"

        if elbow_angle < 120:  # Low plank
            if elbow_angle < 70 or elbow_angle > 110:
                potential_errors["Elbow angle incorrect"] = f"Elbow angle: {elbow_angle:.2f}° not ~90°"
        else:  # High plank
            if elbow_angle < 160:
                potential_errors["Arms not straight"] = f"Elbow angle: {elbow_angle:.2f}° < 160°"

        # Cập nhật thời gian lỗi
        for error, reason in potential_errors.items():
            if error not in error_timestamps:
                error_timestamps[error] = current_time
                print(f"Potential error detected: {error} ({reason})")
            elif current_time - error_timestamps[error] > 0.5:
                confirmed_errors.append(error)
                print(f"Confirmed error: {error} (Duration: {current_time - error_timestamps[error]:.2f}s)")

        # Xóa lỗi nếu không còn tồn tại
        for error in list(error_timestamps.keys()):
            if error not in potential_errors:
                print(f"Error cleared: {error}")
                del error_timestamps[error]

    except Exception as e:
        print(f"Error in check_plank: {e}")

    status = "Correct" if not confirmed_errors else "Incorrect"
    if status == "Correct":
        print("No confirmed errors")
    return status, confirmed_errors

# Xử lý webcam
def analyze_plank(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {source}")
        return

    frame_count = 0
    start_time = time.time()
    error_timestamps = {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Starting webcam, assumed FPS: {fps}")
    print("Tip: Place webcam at side view (90° angle) to see your full body clearly.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to read frame at frame_count {frame_count}, time {time.time() - start_time:.2f}s")
            cap.release()
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print("Error: Could not reopen webcam. Exiting...")
                break
            time.sleep(1)
            continue

        frame_count += 1
        current_time = time.time() - start_time
        if frame_count % 3 == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                status, errors = check_plank(landmarks, error_timestamps, current_time)

                cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if errors:
                    for i, error in enumerate(errors):
                        cv2.putText(frame, f"Error {i+1}: {error}", (10, 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "No errors detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            else:
                print(f"Warning: No landmarks detected at frame {frame_count}")
                cv2.putText(frame, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Plank Analysis", frame)
            elapsed_time = time.time() - start_time
            real_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            print(f"Processed frame {frame_count}, Time: {elapsed_time:.2f}s, Real FPS: {real_fps:.2f}")

        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            print("User stopped the program")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program ended")

# Chạy
if __name__ == "__main__":
    source = r"C:\Users\fox\Downloads\202504032106.mp4"  # Webcam
    analyze_plank(source)