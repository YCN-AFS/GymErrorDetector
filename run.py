import cv2
import mediapipe as mp
import pickle
import pandas as pd
import numpy as np
import warnings
import time
from scipy.spatial import distance
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# warnings.filterwarnings("ignore")
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load mô hình đã huấn luyện
with open('pose_model.pkl', 'rb') as f:
    model = pickle.load(f)

columns = ('x1', 'y1', 'z1', 'v1', 'x2', 'y2', 'z2', 'v2', 'x3', 'y3', 'z3', 'v3', 'x4', 'y4', 'z4', 'v4',
           'x5', 'y5', 'z5', 'v5', 'x6', 'y6', 'z6', 'v6', 'x7', 'y7', 'z7', 'v7', 'x8', 'y8', 'z8', 'v8',
           'x9', 'y9', 'z9', 'v9', 'x10', 'y10', 'z10', 'v10', 'x11', 'y11', 'z11', 'v11', 'x12', 'y12', 'z12', 'v12',
           'x13', 'y13', 'z13', 'v13', 'x14', 'y14', 'z14', 'v14', 'x15', 'y15', 'z15', 'v15', 'x16', 'y16', 'z16',
           'v16',
           'x17', 'y17', 'z17', 'v17', 'x18', 'y18', 'z18', 'v18', 'x19', 'y19', 'z19', 'v19', 'x20', 'y20', 'z20',
           'v20',
           'x21', 'y21', 'z21', 'v21', 'x22', 'y22', 'z22', 'v22', 'x23', 'y23', 'z23', 'v23', 'x24', 'y24', 'z24',
           'v24',
           'x25', 'y25', 'z25', 'v25', 'x26', 'y26', 'z26', 'v26', 'x27', 'y27', 'z27', 'v27', 'x28', 'y28', 'z28',
           'v28',
           'x29', 'y29', 'z29', 'v29', 'x30', 'y30', 'z30', 'v30', 'x31', 'y31', 'z31', 'v31', 'x32', 'y32', 'z32',
           'v32',
           'x33', 'y33', 'z33', 'v33')


# Initialize the fall detection variables


fall_start_time = None
normal_start_time = None
fall_detected = False

# cap = cv2.VideoCapture(0)

def find_available_camera(max_index=10):
    cap = None
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            return cap
        else:
            cap.release()
    print("No available camera found.")
    return None

cap = find_available_camera()

cap.set(3, 640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640 * 2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480 * 2)
ret, frame = cap.read()
height, width = frame.shape[:2]


def draw_detection_box(image, landmarks, is_fall):
    # print(f"Is fall: {is_fall}")
    landmark_points = np.array([(landmark.x * width, landmark.y * height) for landmark in landmarks])
    x, y, w, h = cv2.boundingRect(landmark_points.astype(int))
    color = (0, 0, 255) if is_fall else (0, 255, 0)  # Đỏ nếu té ngã, xanh lá nếu bình thường
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)


    status = "Not Good" if is_fall else "Good"
    text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    text_x = x + (w - text_size[0]) // 2
    text_y = y - 10 if y - 10 > text_size[1] else y + text_size[1]
    cv2.putText(image, status, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return x, y, w, h

def calculate_angle(a, b, c):
    """Tính góc giữa ba điểm"""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def load_reference_pose(class_name):
    """Load pose tham chiếu từ file"""
    reference_dir = "reference_poses"
    landmarks_path = os.path.join(reference_dir, f"{class_name}_landmarks.npy")
    
    if os.path.exists(landmarks_path):
        return np.load(landmarks_path)
    return None

def analyze_pose_errors(landmarks, reference_landmarks):
    """Phân tích lỗi tư thế chi tiết"""
    errors = []
    
    try:
        # Kiểm tra góc khuỷu tay phải
        right_elbow_angle = calculate_angle(
            landmarks[11], landmarks[13], landmarks[15]
        )
        ref_right_elbow = calculate_angle(
            reference_landmarks[11], reference_landmarks[13], reference_landmarks[15]
        )
        if abs(right_elbow_angle - ref_right_elbow) > 15:
            errors.append("Góc khuỷu tay phải chưa đúng")
        
        # Kiểm tra góc khuỷu tay trái
        left_elbow_angle = calculate_angle(
            landmarks[12], landmarks[14], landmarks[16]
        )
        ref_left_elbow = calculate_angle(
            reference_landmarks[12], reference_landmarks[14], reference_landmarks[16]
        )
        if abs(left_elbow_angle - ref_left_elbow) > 15:
            errors.append("Góc khuỷu tay trái chưa đúng")
        
        # Kiểm tra góc đầu gối phải
        right_knee_angle = calculate_angle(
            landmarks[23], landmarks[25], landmarks[27]
        )
        ref_right_knee = calculate_angle(
            reference_landmarks[23], reference_landmarks[25], reference_landmarks[27]
        )
        if abs(right_knee_angle - ref_right_knee) > 15:
            errors.append("Góc đầu gối phải chưa đúng")
        
        # Kiểm tra góc đầu gối trái
        left_knee_angle = calculate_angle(
            landmarks[24], landmarks[26], landmarks[28]
        )
        ref_left_knee = calculate_angle(
            reference_landmarks[24], reference_landmarks[26], reference_landmarks[28]
        )
        if abs(left_knee_angle - ref_left_knee) > 15:
            errors.append("Góc đầu gối trái chưa đúng")
        
        # Kiểm tra tư thế cột sống
        spine_angle = calculate_angle(
            landmarks[11], landmarks[23], landmarks[25]
        )
        ref_spine_angle = calculate_angle(
            reference_landmarks[11], reference_landmarks[23], reference_landmarks[25]
        )
        if abs(spine_angle - ref_spine_angle) > 15:
            errors.append("Tư thế cột sống chưa đúng")
            
    except Exception as e:
        print(f"Error in analyze_pose_errors: {e}")
        
    return errors

def draw_feedback(image, errors, box_coords):
    """Hiển thị phản hồi chi tiết trên màn hình"""
    x, y, w, h = box_coords
    
    # Vẽ khung chứa phản hồi
    cv2.rectangle(image, (x-10, y+h+25), (x+w+10, y+h+60), (0,0,0), -1)
    
    # Hiển thị các lỗi
    y_offset = y+h+45
    for error in errors:
        cv2.putText(image, error, (x, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        y_offset += 20

class PoseEvaluator:
    def __init__(self, model_path):
        self.model = model
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def calculate_joint_angles(self, landmarks):
        # Tính toán các góc khớp quan trọng
        # Ví dụ: góc khuỷu tay, góc đầu gối...
        angles = {}
        # TODO: Thêm logic tính góc ở đây
        return angles
    
    def compare_with_standard(self, current_pose, standard_pose):
        # So sánh sử dụng DTW
        distance = fastdtw(current_pose, standard_pose)
        return distance
    
    def evaluate_realtime(self):
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image)
            
            if results.pose_landmarks:
                # Vẽ các điểm landmark
                self.mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
                
                # Trích xuất đặc trưng
                pose_landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    pose_landmarks.extend([landmark.x, landmark.y, landmark.z])
                
                # Đánh giá động tác
                prediction = self.model.predict_proba([pose_landmarks])[0]
                
                # Hiển thị kết quả
                cv2.putText(
                    image,
                    f"Score: {prediction[0]:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            
            # Hiển thị frame
            cv2.imshow('Pose Evaluation', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(5) & 0xFF == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    evaluator = PoseEvaluator('path_to_saved_model.h5')
    evaluator.evaluate_realtime()