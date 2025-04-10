import cv2
import mediapipe as mp
import os
import argparse
import glob
from pathlib import Path
import subprocess
import tempfile
import shutil

# Khởi tạo pose và drawing utils
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
from mediapipe.framework.formats import landmark_pb2

# Danh sách ID các điểm trên mặt trong pose
FACE_LANDMARK_IDS = list(range(0, 11))

# Định nghĩa màu và độ dày riêng cho các connections
CUSTOM_CONNECTIONS_STYLE = {
    # Đùi trái (LEFT_HIP (23) đến LEFT_KNEE (25))
    # (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE): mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4),
    
    # Đùi phải (RIGHT_HIP (24) đến RIGHT_KNEE (26))
    # (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE): mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4),
    
    # Mông (LEFT_HIP (23) đến RIGHT_HIP (24))
    # (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP): mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=4),
}

# Định nghĩa các phần mở rộng file
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

def process_image(image_path, output_path):
    """Xử lý pose trên ảnh"""
    # Đọc ảnh và chuyển sang RGB
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh từ {image_path}")
        return False
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Xử lý pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0) as pose:
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            # Tạo bản sao ảnh
            annotated_image = image.copy()

            # Loại bỏ các điểm trên mặt
            new_landmarks = landmark_pb2.NormalizedLandmarkList()
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                if idx in FACE_LANDMARK_IDS:
                    invisible = landmark_pb2.NormalizedLandmark(x=0, y=0, z=0, visibility=0)
                    new_landmarks.landmark.append(invisible)
                else:
                    new_landmarks.landmark.append(lm)

            # Vẽ toàn bộ các connections
            draw_connections(annotated_image, new_landmarks)

            # Đảm bảo thư mục đầu ra tồn tại
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Lưu kết quả
            cv2.imwrite(output_path, annotated_image)
            print(f"Đã lưu ảnh có pose (không vẽ mặt) tại: {output_path}")
            return True
        else:
            print(f"Không phát hiện pose trong ảnh: {image_path}")
            return False

def process_video(video_path, output_path):
    """Xử lý pose trên video và giữ nguyên âm thanh"""
    # Tạo tên file tạm thời cho video không có âm thanh
    temp_dir = tempfile.gettempdir()
    temp_video = os.path.join(temp_dir, f"temp_{os.path.basename(output_path)}")
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Không thể mở video từ {video_path}")
        return False
    
    # Lấy thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Tạo video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec hoạt động với hầu hết các hệ thống
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    
    # Xử lý từng frame
    with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5) as pose:
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Hiển thị tiến trình
            frame_count += 1
            if frame_count % 30 == 0 or frame_count == 1:
                print(f"Video {os.path.basename(video_path)}: Đang xử lý frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
            
            # Xử lý frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Loại bỏ các điểm trên mặt
                new_landmarks = landmark_pb2.NormalizedLandmarkList()
                for idx, lm in enumerate(results.pose_landmarks.landmark):
                    if idx in FACE_LANDMARK_IDS:
                        invisible = landmark_pb2.NormalizedLandmark(x=0, y=0, z=0, visibility=0)
                        new_landmarks.landmark.append(invisible)
                    else:
                        new_landmarks.landmark.append(lm)
                        
                # Vẽ các connections
                draw_connections(frame, new_landmarks)
            
            # Ghi frame đã xử lý
            out.write(frame)
            
    # Giải phóng tài nguyên
    cap.release()
    out.release()
    
    # Kết hợp video và âm thanh bằng FFmpeg
    try:
        # Kiểm tra xem FFmpeg có được cài đặt không
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        # Sử dụng FFmpeg để kết hợp video đã xử lý với âm thanh từ video gốc
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_video,    # Video đã xử lý (không có âm thanh)
            "-i", video_path,    # Video gốc (lấy âm thanh)
            "-c:v", "copy",      # Giữ nguyên video
            "-c:a", "aac",       # Sao chép âm thanh
            "-map", "0:v:0",     # Lấy video từ tệp đầu tiên
            "-map", "1:a:0?",    # Lấy âm thanh từ tệp thứ hai nếu có
            "-shortest",         # Kết thúc khi luồng ngắn nhất kết thúc
            output_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"Đã lưu video có pose (giữ nguyên âm thanh) tại: {output_path}")
        
        # Xóa tệp tạm thời
        if os.path.exists(temp_video):
            os.remove(temp_video)
        
        return True
    except FileNotFoundError:
        # Nếu không có FFmpeg, chỉ đổi tên file tạm
        print("Không tìm thấy FFmpeg. Video sẽ được lưu mà không có âm thanh.")
        shutil.move(temp_video, output_path)
        print(f"Đã lưu video có pose (không có âm thanh) tại: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        # Nếu FFmpeg gặp lỗi
        print(f"Lỗi khi kết hợp âm thanh: {e}")
        print("Sử dụng video không có âm thanh...")
        shutil.move(temp_video, output_path)
        print(f"Đã lưu video có pose (không có âm thanh) tại: {output_path}")
        return True

def draw_connections(image, landmarks):
    """Vẽ các đường kết nối giữa các điểm landmark"""
    h, w, _ = image.shape
    
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        start_lm = landmarks.landmark[start_idx]
        end_lm = landmarks.landmark[end_idx]

        # Bỏ qua nếu điểm không hiển thị
        if start_lm.visibility < 0.5 or end_lm.visibility < 0.5:
            continue

        start_point = (int(start_lm.x * w), int(start_lm.y * h))
        end_point = (int(end_lm.x * w), int(end_lm.y * h))

        # Kiểm tra xem có dùng style riêng không
        if (start_idx, end_idx) in CUSTOM_CONNECTIONS_STYLE:
            style = CUSTOM_CONNECTIONS_STYLE[(start_idx, end_idx)]
            cv2.line(image, start_point, end_point, style.color, style.thickness)
        elif (end_idx, start_idx) in CUSTOM_CONNECTIONS_STYLE:
            style = CUSTOM_CONNECTIONS_STYLE[(end_idx, start_idx)]
            cv2.line(image, start_point, end_point, style.color, style.thickness)
        else:
            # Dùng màu vàng hiện đại cho các connections còn lại
            cv2.line(image, start_point, end_point, (0, 255, 0), 6)

def process_directory(input_dir, output_dir):
    """Xử lý tất cả ảnh và video trong thư mục"""
    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # Đếm số lượng file đã xử lý
    processed_files = 0
    total_files = 0
    
    # Đếm tổng số file cần xử lý
    for ext in IMAGE_EXTENSIONS + VIDEO_EXTENSIONS:
        pattern = os.path.join(input_dir, f"*{ext}")
        total_files += len(glob.glob(pattern, recursive=True))
        pattern = os.path.join(input_dir, f"*{ext.upper()}")
        total_files += len(glob.glob(pattern, recursive=True))
    
    print(f"Tìm thấy {total_files} file để xử lý...")
    
    # Xử lý tất cả ảnh
    for ext in IMAGE_EXTENSIONS:
        # Tìm cả chữ thường và chữ hoa
        for pattern in [f"*{ext}", f"*{ext.upper()}"]:
            for image_path in glob.glob(os.path.join(input_dir, pattern)):
                # Tạo đường dẫn đầu ra
                rel_path = os.path.relpath(image_path, input_dir)
                output_path = os.path.join(output_dir, f"{os.path.splitext(rel_path)[0]}_pose.jpg")
                
                # Xử lý ảnh
                processed_files += 1
                print(f"[{processed_files}/{total_files}] Đang xử lý ảnh: {rel_path}")
                process_image(image_path, output_path)
    
    # Xử lý tất cả video
    for ext in VIDEO_EXTENSIONS:
        # Tìm cả chữ thường và chữ hoa
        for pattern in [f"*{ext}", f"*{ext.upper()}"]:
            for video_path in glob.glob(os.path.join(input_dir, pattern)):
                # Tạo đường dẫn đầu ra
                rel_path = os.path.relpath(video_path, input_dir)
                output_path = os.path.join(output_dir, f"{os.path.splitext(rel_path)[0]}_pose.mp4")
                
                # Xử lý video với âm thanh
                processed_files += 1
                print(f"[{processed_files}/{total_files}] Đang xử lý video: {rel_path}")
                process_video(video_path, output_path)
    
    print(f"Hoàn thành! Đã xử lý {processed_files} file, kết quả được lưu tại: {output_dir}")

def main():
    # Phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description='Xử lý pose từ thư mục ảnh và video')
    parser.add_argument('--input_dir', type=str, required=True, help='Thư mục chứa ảnh và video đầu vào')
    parser.add_argument('--output_dir', type=str, required=True, help='Thư mục lưu kết quả')
    
    args = parser.parse_args()
    
    # Kiểm tra thư mục đầu vào
    if not os.path.isdir(args.input_dir):
        print(f"Thư mục đầu vào không tồn tại: {args.input_dir}")
        return
    
    # Xử lý thư mục
    process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()