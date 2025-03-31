import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import time
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image, ImageDraw, ImageFont
import io

# Thêm biến global
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def draw_emoji(image: np.ndarray, emoji: str, position: Tuple[int, int], size: int = 30) -> np.ndarray:
    """Draw emoji on OpenCV image using PIL."""
    # Convert OpenCV image to PIL Image
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # Load emoji font
    try:
        font = ImageFont.truetype("arial.ttf", size)
    except:
        font = ImageFont.load_default()
    
    # Draw emoji
    draw.text(position, emoji, font=font)
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

class PoseLSTM(nn.Module):
    def __init__(self, input_size):
        super(PoseLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = x[:, -1, :]
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.bn2(x)
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class PoseVisualizer:
    def __init__(self):
        self.status_box_pos = None
        self.feedback_box_pos = None
        self.is_dragging_status = False
        self.is_dragging_feedback = False
        self.drag_offset = (0, 0)
        self.fps_list = []
        self.start_time = time.time()
        self.frame_count = 0

    def draw_pose(self, frame: np.ndarray, landmarks: Any, score: float, label: str, fps: float) -> np.ndarray:
        """Draw pose landmarks and evaluation results on frame."""
        try:
            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                frame,
                landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            # Draw UI elements
            frame = self.draw_status_box(frame, score, label, fps)
            frame = self.draw_feedback_details(frame, score)
            frame = self.draw_joint_angles(frame, landmarks)

            return frame
        except Exception as e:
            print(f"Error drawing pose: {e}")
            return frame

    def draw_status_box(self, image: np.ndarray, score: float, label: str, fps: float) -> np.ndarray:
        """Draw a draggable status box with performance metrics."""
        h, w = image.shape[:2]
        margin = 20
        box_width = 300
        box_height = 130
        
        # Initialize default position if not set
        if self.status_box_pos is None:
            self.status_box_pos = (w - box_width - margin, margin)
        
        # Draw semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, 
                     (self.status_box_pos[0], self.status_box_pos[1]),
                     (self.status_box_pos[0] + box_width, self.status_box_pos[1] + box_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw border
        cv2.rectangle(image,
                     (self.status_box_pos[0], self.status_box_pos[1]),
                     (self.status_box_pos[0] + box_width, self.status_box_pos[1] + box_height),
                     (255, 255, 255), 2)
        
        # Draw progress bar
        progress_width = int(score * (box_width - 40))
        cv2.rectangle(image,
                     (self.status_box_pos[0] + 20, self.status_box_pos[1] + 70),
                     (self.status_box_pos[0] + box_width - 20, self.status_box_pos[1] + 90),
                     (100, 100, 100), -1)
        cv2.rectangle(image,
                     (self.status_box_pos[0] + 20, self.status_box_pos[1] + 70),
                     (self.status_box_pos[0] + 20 + progress_width, self.status_box_pos[1] + 90),
                     (0, 255, 0) if score > 0.5 else (0, 0, 255), -1)
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # FPS counter
        cv2.putText(image, f"FPS: {fps:.1f}",
                    (self.status_box_pos[0] + 20, self.status_box_pos[1] + 25),
                    font, 0.6, (255, 255, 255), 1)
        
        # Status text
        status_text = f"Status: {label}"
        text_size = cv2.getTextSize(status_text, font, 0.7, 2)[0]
        text_x = self.status_box_pos[0] + (box_width - text_size[0]) // 2
        cv2.putText(image, status_text,
                    (text_x, self.status_box_pos[1] + 50),
                    font, 0.7, (255, 255, 255), 2)
        
        # Confidence score
        conf_text = f"Confidence: {score:.2%}"
        conf_size = cv2.getTextSize(conf_text, font, 0.7, 2)[0]
        conf_x = self.status_box_pos[0] + (box_width - conf_size[0]) // 2
        cv2.putText(image, conf_text,
                    (conf_x, self.status_box_pos[1] + 110),
                    font, 0.7, (255, 255, 255), 2)

        return image

    def draw_feedback_details(self, image: np.ndarray, score: float) -> np.ndarray:
        """Draw detailed feedback box with suggestions."""
        h, w = image.shape[:2]
        feedback_text = []
        
        if score <= 0.5:
            feedback_text = [
                "[!] Exercise Feedback:",
                "> Check posture alignment",
                "> Maintain proper balance",
                "> Follow movement rhythm",
                "> Keep consistent speed",
                "> Focus on form"
            ]
        else:
            feedback_text = [
                "[+] Great Job!",
                "> Perfect posture",
                "> Excellent balance",
                "> Proper movement",
                "> Consistent pace",
                "> Keep it up!"
            ]
        
        # Initialize default position if not set
        if self.feedback_box_pos is None:
            self.feedback_box_pos = (20, 20)
        
        text_height = len(feedback_text) * 25 + 20
        
        # Draw semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay,
                     (self.feedback_box_pos[0], self.feedback_box_pos[1]),
                     (self.feedback_box_pos[0] + 280, self.feedback_box_pos[1] + text_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Draw border
        cv2.rectangle(image,
                     (self.feedback_box_pos[0], self.feedback_box_pos[1]),
                     (self.feedback_box_pos[0] + 280, self.feedback_box_pos[1] + text_height),
                     (255, 255, 255), 2)
        
        # Draw text with bullet points
        for i, text in enumerate(feedback_text):
            color = (0, 255, 0) if score > 0.5 else (0, 165, 255)
            cv2.putText(image, text,
                       (self.feedback_box_pos[0] + 10, self.feedback_box_pos[1] + 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return image

    def draw_joint_angles(self, image: np.ndarray, landmarks: Any) -> np.ndarray:
        """Draw joint angles with visual indicators."""
        h, w = image.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Define joint pairs and their colors
        joint_pairs = [
            ('R_Elbow', (mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                        mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                        mp_pose.PoseLandmark.RIGHT_WRIST.value), (0, 255, 0)),
            ('L_Elbow', (mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                        mp_pose.PoseLandmark.LEFT_ELBOW.value,
                        mp_pose.PoseLandmark.LEFT_WRIST.value), (0, 255, 0)),
            ('R_Knee', (mp_pose.PoseLandmark.RIGHT_HIP.value,
                       mp_pose.PoseLandmark.RIGHT_KNEE.value,
                       mp_pose.PoseLandmark.RIGHT_ANKLE.value), (255, 0, 0)),
            ('L_Knee', (mp_pose.PoseLandmark.LEFT_HIP.value,
                       mp_pose.PoseLandmark.LEFT_KNEE.value,
                       mp_pose.PoseLandmark.LEFT_ANKLE.value), (255, 0, 0))
        ]
        
        for name, (p1, p2, p3), color in joint_pairs:
            angle = self._calculate_angle(
                landmarks[p1],
                landmarks[p2],
                landmarks[p3]
            )
            
            # Draw joint point
            x = int(landmarks[p2].x * w)
            y = int(landmarks[p2].y * h)
            cv2.circle(image, (x, y), 5, color, -1)
            
            # Draw angle arc
            radius = 20
            start_angle = int(angle)
            cv2.ellipse(image, (x, y), (radius, radius), 0, 0, start_angle, color, 2)
            
            # Draw angle text
            angle_text = f'{name}: {angle:.1f}°'
            cv2.putText(image, angle_text, (x-10, y-10),
                       font, 0.5, color, 2)
        
        return image

    def _calculate_angle(self, a: Any, b: Any, c: Any) -> float:
        """Calculate angle between three points."""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)

    def handle_mouse_event(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Handle mouse events for dragging UI elements."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._point_in_box(x, y, self.status_box_pos, 300, 130):
                self.is_dragging_status = True
                self.drag_offset = (x - self.status_box_pos[0], y - self.status_box_pos[1])
            elif self._point_in_box(x, y, self.feedback_box_pos, 280, 170):
                self.is_dragging_feedback = True
                self.drag_offset = (x - self.feedback_box_pos[0], y - self.feedback_box_pos[1])
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging_status:
                self.status_box_pos = (x - self.drag_offset[0], y - self.drag_offset[1])
            elif self.is_dragging_feedback:
                self.feedback_box_pos = (x - self.drag_offset[0], y - self.drag_offset[1])
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging_status = False
            self.is_dragging_feedback = False

    def _point_in_box(self, x: int, y: int, box_pos: Tuple[int, int], width: int, height: int) -> bool:
        """Check if a point is inside a box."""
        return (box_pos[0] <= x <= box_pos[0] + width and 
                box_pos[1] <= y <= box_pos[1] + height)

def predict_pose():
    # Khởi tạo MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Load mô hình đã huấn luyện
    input_size = 132
    model = PoseLSTM(input_size)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # Khởi tạo visualizer
    visualizer = PoseVisualizer()

    # Mở video
    cap = cv2.VideoCapture(r"C:\Users\fox\Downloads\202502191737.mp4")
    
    # Lấy kích thước màn hình
    screen_width = 1280
    screen_height = 720

    fps_list = []
    start_time = time.time()
    frame_count = 0

    window_name = 'Gym Quality Review'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, visualizer.handle_mouse_event)

    while cap.isOpened():
        frame_start_time = time.time()
        success, image = cap.read()
        if not success:
            continue

        # Điều chỉnh kích thước frame
        h, w = image.shape[:2]
        ratio = min(screen_width/w, screen_height/h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Xử lý frame
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # Trích xuất đặc trưng
            pose_landmarks = []
            for landmark in results.pose_landmarks.landmark:
                pose_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

            # Dự đoán và hiển thị
            pose_tensor = torch.FloatTensor(pose_landmarks).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                prediction = model(pose_tensor)
                score = prediction.item()

            # Tính FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - start_time)
                fps_list.append(fps)
                start_time = time.time()

            current_fps = np.mean(fps_list[-10:]) if fps_list else 0
            
            # Vẽ giao diện
            label = "Good Form" if score > 0.5 else "Need Improvement"
            image = visualizer.draw_pose(image, results.pose_landmarks, score, label, current_fps)

        # Hiển thị frame
        cv2.imshow(window_name, image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Thời gian xử lý trung bình: {np.mean(fps_list):.4f} giây/frame")
    print(f"FPS trung bình: {current_fps:.2f}")

if __name__ == "__main__":
    predict_pose() 