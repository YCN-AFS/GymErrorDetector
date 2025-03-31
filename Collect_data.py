import os
import csv
import winsound 
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import mediapipe as mp
import threading
import time
import pandas as pd
import shutil

class PoseDataCollectorApp:
    def __init__(self, master):
        self.master = master
        master.title("Pose Data Collector")
        master.geometry("600x500")

        # Setup MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.FILE_NAME = "video_poses.csv"
        self.default_duration = 20

        # Create UI components
        self.create_widgets()

        # Camera capture variables
        self.is_capturing = False
        self.capture_thread = None

    def create_widgets(self):
        # Frame for buttons
        button_frame = tk.Frame(self.master)
        button_frame.pack(pady=20)

        # Camera Capture Button
        camera_btn = tk.Button(button_frame, text="Capture from Camera", command=self.open_camera_dialog)
        camera_btn.pack(side=tk.LEFT, padx=10)

        # Single Video File Button
        video_file_btn = tk.Button(button_frame, text="Process Video File", command=self.process_single_video)
        video_file_btn.pack(side=tk.LEFT, padx=10)

        # Video Folder Processing Button
        video_btn = tk.Button(button_frame, text="Process Video Folder", command=self.process_video_folder)
        video_btn.pack(side=tk.LEFT, padx=10)

        # Clear Data Button
        clear_btn = tk.Button(button_frame, text="Clear Data", command=self.clear_data)
        clear_btn.pack(side=tk.LEFT, padx=10)

        # Logging Text Area
        self.log_text = tk.Text(self.master, height=15, width=70)
        self.log_text.pack(pady=20)

        # Status Label
        self.status_label = tk.Label(self.master, text="Ready", fg="green")
        self.status_label.pack(pady=10)

    def log_message(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.status_label.config(text=message, fg="green")

    def open_camera_dialog(self):
        # Hộp thoại nhập danh sách class
        class_names_str = simpledialog.askstring("Input", "Enter action/class names (separated by comma):")
        if not class_names_str:
            return
        
        # Chuyển string thành list các class
        class_names = [name.strip() for name in class_names_str.split(',')]
        
        # Nhập các thông số chung
        camera_id = simpledialog.askinteger("Input", "Enter camera ID (default 0):", 
                                          initialvalue=0, minvalue=0)
        if camera_id is None:
            return
        
        duration = simpledialog.askinteger("Input", 
                                         f"Enter recording duration in seconds (default {self.default_duration}):", 
                                         initialvalue=self.default_duration, 
                                         minvalue=1)
        if duration is None:
            duration = self.default_duration

        # Tạo thread mới để xử lý toàn bộ quá trình
        processing_thread = threading.Thread(
            target=self.process_all_classes,
            args=(class_names, camera_id, duration)
        )
        processing_thread.start()

    def process_all_classes(self, class_names, camera_id, duration):
        for class_name in class_names:
            self.countdown_and_record(class_name, camera_id, duration)

    def countdown_and_record(self, class_name, camera_id, duration):
        # Hiển thị thông báo chuẩn bị
        self.log_message(f"Preparing to record {class_name}")
        self.log_message("Get ready in:")
        
        # Đếm ngược 5 giây
        for i in range(5, 0, -1):
            self.log_message(str(i))
            winsound.Beep(1000, 500)
            time.sleep(0.5)
        
        # Phát âm thanh bắt đầu
        winsound.Beep(2000, 1000)
        self.log_message("START RECORDING!")
        
        # Đảm bảo trạng thái capturing được reset
        self.is_capturing = False
        time.sleep(0.5)  # Đợi một chút để đảm bảo trạng thái đã được reset
        
        # Bắt đầu thu thập
        self.process_from_camera(class_name, camera_id, duration)
        
        # Đợi thêm 2 giây trước khi bắt đầu class tiếp theo
        time.sleep(2)

    def process_from_camera(self, class_name, camera_id=0, duration=None):
        if duration is None:
            duration = self.default_duration
        
        self.is_capturing = True
        first_time = not os.path.exists(self.FILE_NAME)
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            self.log_message("Error: Could not open camera")
            self.is_capturing = False
            return
        
        self.log_message(f"Recording data for action: {class_name}")
        self.log_message(f"Recording for {duration} seconds. Press 'q' to stop early.")
        
        start_time = cv2.getTickCount()
        
        with self.mp_pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5, 
            model_complexity=2
        ) as pose:
            while cap.isOpened() and self.is_capturing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                if current_time >= duration:
                    self.log_message(f"Completed recording {class_name}!")
                    break
                
                time_left = int(duration - current_time)
                cv2.putText(frame, f'Recording: {class_name}', (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Time left: {time_left}s', (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                
                results = pose.process(image_rgb)
                
                image_rgb.flags.writeable = True
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                try:
                    body_pose = results.pose_landmarks.landmark
                    num_coords = len(body_pose)
                    
                    if first_time:
                        mark = ['class']
                        for val in range(1, num_coords + 1):
                            mark += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']
                        with open(self.FILE_NAME, 'w', newline='', encoding='utf-8') as csvfile:
                            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            writer.writerow(mark)
                        first_time = False
                    
                    pose_row = list(np.array(
                        [[landmarks.x, landmarks.y, landmarks.z, landmarks.visibility] for landmarks in body_pose]).flatten())
                    pose_row.insert(0, class_name)
                    
                    with open(self.FILE_NAME, 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        writer.writerow(pose_row)
                        
                except Exception as e:
                    self.log_message(f"Error processing frame: {e}")
                
                self.mp_drawing.draw_landmarks(
                    image_bgr, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS
                )
                cv2.imshow('Pose Estimation', image_bgr)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        self.is_capturing = False
        self.log_message(f"Finished recording {class_name}")

    def process_video_folder(self):
        video_dir = filedialog.askdirectory(title="Select Video Folder")
        if video_dir:
            threading.Thread(target=self.process_videos_in_directory, args=(video_dir,)).start()

    def process_videos_in_directory(self, video_dir):
        action_dirs = [
            d for d in os.listdir(video_dir) 
            if os.path.isdir(os.path.join(video_dir, d))
        ]
        
        for action in action_dirs:
            action_path = os.path.join(video_dir, action)
            video_files = [
                f for f in os.listdir(action_path) 
                if f.endswith(('.mp4', '.avi', '.mov'))
            ]
            
            self.log_message(f"Processing action: {action}")
            for video_file in video_files:
                video_path = os.path.join(action_path, video_file)
                self.log_message(f"Processing video: {video_file}")
                self.process_video(video_path, action)

    def process_video(self, video_path, class_name):
        first_time = not os.path.exists(self.FILE_NAME)
        cap = cv2.VideoCapture(video_path)
        
        with self.mp_pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5, 
            model_complexity=2
        ) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                
                results = pose.process(image_rgb)
                
                try:
                    body_pose = results.pose_landmarks.landmark
                    num_coords = len(body_pose)
                    
                    if first_time:
                        mark = ['class']
                        for val in range(1, num_coords + 1):
                            mark += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']
                        with open(self.FILE_NAME, 'w', newline='', encoding='utf-8') as csvfile:
                            writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            writer.writerow(mark)
                        first_time = False
                    
                    pose_row = list(np.array(
                        [[landmarks.x, landmarks.y, landmarks.z, landmarks.visibility] for landmarks in body_pose]).flatten())
                    pose_row.insert(0, class_name)
                    
                    with open(self.FILE_NAME, 'a', newline='', encoding='utf-8') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        writer.writerow(pose_row)
                        
                except Exception as e:
                    self.log_message(f"Error processing video: {e}")
                
        cap.release()

    def clear_data(self):
        if os.path.exists(self.FILE_NAME):
            os.remove(self.FILE_NAME)
            self.log_message("Cleared all data!")
        else:
            self.log_message("No data file to clear!")

    def process_single_video(self):
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        if video_path:
            class_name = simpledialog.askstring("Input", "Enter action/class name:")
            if class_name:
                threading.Thread(target=self.process_video, args=(video_path, class_name)).start()
                self.log_message(f"Processing video: {os.path.basename(video_path)}")

    def save_reference_pose(self, class_name, video_path):
        """Lưu video tham chiếu cho một động tác"""
        reference_dir = "reference_poses"
        if not os.path.exists(reference_dir):
            os.makedirs(reference_dir)
        
        # Copy video tham chiếu
        ref_path = os.path.join(reference_dir, f"{class_name}_reference.mp4")
        shutil.copy2(video_path, ref_path)
        
        # Trích xuất và lưu landmarks
        landmarks = []
        cap = cv2.VideoCapture(video_path)
        
        with self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    frame_landmarks = []
                    for landmark in results.pose_landmarks.landmark:
                        frame_landmarks.extend([landmark.x, landmark.y, landmark.z])
                    landmarks.append(frame_landmarks)
        
        cap.release()
        
        # Lưu landmarks tham chiếu
        np.save(os.path.join(reference_dir, f"{class_name}_landmarks.npy"), 
                np.array(landmarks))

class PoseDataCollector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def extract_landmarks(self, frame):
        # Chuyển đổi màu BGR sang RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        landmarks = []
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
        
        return landmarks
    
    def calculate_angles(self, landmarks):
        # Tính toán các góc khớp quan trọng
        angles = {}
        # Ví dụ: góc khuỷu tay
        if len(landmarks) >= 33*4:  # Đảm bảo đủ landmarks
            shoulder = np.array([landmarks[11*4], landmarks[11*4+1]])
            elbow = np.array([landmarks[13*4], landmarks[13*4+1]])
            wrist = np.array([landmarks[15*4], landmarks[15*4+1]])
            
            angle = self.calculate_angle(shoulder, elbow, wrist)
            angles['elbow'] = angle
            
        return angles
    
    @staticmethod
    def calculate_angle(a, b, c):
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    
    def collect_data(self, video_path, output_csv):
        cap = cv2.VideoCapture(video_path)
        data = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            landmarks = self.extract_landmarks(frame)
            if landmarks:
                angles = self.calculate_angles(landmarks)
                data.append(landmarks + list(angles.values()))
        
        cap.release()
        
        # Lưu dữ liệu vào CSV
        columns = [f'landmark_{i}_{coord}' for i in range(33) for coord in ['x','y','z','v']]
        columns.extend(['angle_elbow'])  # Thêm các cột góc
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(output_csv, index=False)

def main():
    root = tk.Tk()
    app = PoseDataCollectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()