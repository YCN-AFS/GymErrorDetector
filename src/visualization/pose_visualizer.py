import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import time

class PoseVisualizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.status_box_pos = None
        self.feedback_box_pos = None
        self.is_dragging_status = False
        self.is_dragging_feedback = False
        self.drag_offset = (0, 0)
        self.fps_list = []
        self.start_time = time.time()
        self.frame_count = 0

    def draw_pose(self, frame: np.ndarray, landmarks: Any, evaluation: Dict[str, float]) -> np.ndarray:
        """Draw pose landmarks and evaluation results on frame."""
        try:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            # Calculate FPS
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                fps = 30 / (time.time() - self.start_time)
                self.fps_list.append(fps)
                self.start_time = time.time()

            current_fps = np.mean(self.fps_list[-10:]) if self.fps_list else 0

            # Draw UI elements
            frame = self.draw_status_box(frame, evaluation.get('score', 0), 
                                       evaluation.get('label', 'Unknown'), current_fps)
            frame = self.draw_feedback_details(frame, evaluation.get('score', 0))
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
                "📝 Exercise Feedback:",
                "• Check posture alignment",
                "• Maintain proper balance",
                "• Follow movement rhythm",
                "• Keep consistent speed",
                "• Focus on form"
            ]
        else:
            feedback_text = [
                "✅ Great Job!",
                "• Perfect posture",
                "• Excellent balance",
                "• Proper movement",
                "• Consistent pace",
                "• Keep it up!"
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
        
        # Draw text with emojis and bullet points
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

    def draw_fall_warning(self, frame: np.ndarray) -> np.ndarray:
        """Draw fall warning on frame."""
        cv2.putText(
            frame,
            "FALL DETECTED!",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )
        return frame 