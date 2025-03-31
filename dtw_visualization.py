import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import mediapipe as mp
import cv2
from matplotlib.patches import ConnectionPatch
from tqdm import tqdm

def extract_pose_sequence(video_path, max_frames=300):
    """Trích xuất chuỗi pose từ video với giới hạn frame"""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1  # Giảm độ phức tạp của model
    )
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(1, total_frames // max_frames)
    sequence = []
    
    # Tạo thanh tiến trình
    pbar = tqdm(total=min(total_frames, max_frames), desc=f"Processing {video_path.split('/')[-1]}")
    
    frame_count = 0
    while cap.isOpened() and len(sequence) < max_frames:
        success, image = cap.read()
        if not success:
            break
            
        # Bỏ qua frames để tăng tốc độ xử lý
        if frame_count % frame_step != 0:
            frame_count += 1
            continue
            
        # Resize image để tăng tốc độ xử lý
        image = cv2.resize(image, (640, 480))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Tắt ghi vào image để tăng hiệu suất
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        
        if results.pose_landmarks:
            # Chỉ lấy các landmark quan trọng
            important_landmarks = [
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_WRIST,
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.RIGHT_ANKLE,
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.LEFT_ANKLE
            ]
            
            frame_poses = []
            landmarks = results.pose_landmarks.landmark
            for landmark_enum in important_landmarks:
                landmark = landmarks[landmark_enum.value]
                frame_poses.extend([landmark.x, landmark.y])
            sequence.append(frame_poses)
            pbar.update(1)
        
        frame_count += 1
        
        # Cho phép thoát bằng 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    pbar.close()
    cap.release()
    
    if len(sequence) == 0:
        raise ValueError("Không thể trích xuất pose từ video")
        
    return np.array(sequence)

def visualize_dtw_alignment(seq1, seq2, path, save_path='dtw_alignment.png'):
    """Vẽ biểu đồ DTW alignment"""
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: DTW Matrix và Path
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    
    # Tính ma trận khoảng cách
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dtw_matrix[i, j] = euclidean(seq1[i], seq2[j])
    
    # Vẽ heatmap
    im = ax1.imshow(dtw_matrix, aspect='auto', cmap='YlOrRd')
    ax1.plot([p[1] for p in path], [p[0] for p in path], 'b-', linewidth=2, label='Warping Path')
    ax1.set_title('DTW Alignment Matrix')
    ax1.set_xlabel('Standard Sequence')
    ax1.set_ylabel('Test Sequence')
    plt.colorbar(im, ax=ax1)
    
    # Plot 2: Sequence Comparison
    ax2 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
    
    # Index cho khuỷu tay phải (x coordinate)
    # 12 landmarks * 2 (x,y) = 24 features
    # Right elbow là landmark thứ 3 trong danh sách important_landmarks
    joint_idx = 4  # (2 * 2) cho x coordinate của khuỷu tay phải
    
    time1 = np.arange(len(seq1))
    time2 = np.arange(len(seq2))
    
    # Trích xuất dữ liệu của một khớp cụ thể
    seq1_joint = [frame[joint_idx] for frame in seq1]
    seq2_joint = [frame[joint_idx] for frame in seq2]
    
    ax2.plot(time1, seq1_joint, 'b-', label='Standard Sequence')
    ax2.plot(time2, seq2_joint, 'r-', label='Test Sequence')
    
    # Vẽ các đường nối giữa các điểm tương ứng
    for (i, j) in path[::5]:
        con = ConnectionPatch(
            xyA=(time1[i], seq1_joint[i]),
            xyB=(time2[j], seq2_joint[j]),
            coordsA="data", coordsB="data",
            axesA=ax2, axesB=ax2,
            color="gray", alpha=0.3
        )
        ax2.add_artist(con)
    
    ax2.set_title('Right Elbow Trajectory Comparison')  # Cập nhật tiêu đề
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Joint Position (x-coordinate)')  # Làm rõ đơn vị đo
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    try:
        # Đường dẫn đến video chuẩn và video test
        standard_video = r"C:\Users\fox\Downloads\VID_20241220_105651.mp4"
        test_video = r"C:\Users\fox\Downloads\6157788975982.mp4"
        
        print("Đang xử lý video chuẩn...")
        standard_sequence = extract_pose_sequence(standard_video)
        print("Đang xử lý video test...")
        test_sequence = extract_pose_sequence(test_video)
        
        print("Đang tính toán DTW...")
        distance, path = fastdtw(standard_sequence, test_sequence, dist=euclidean)
        
        print("Đang tạo biểu đồ...")
        visualize_dtw_alignment(standard_sequence, test_sequence, path)
        
        print(f"DTW Distance: {distance}")
        print("Hoàn thành! Biểu đồ đã được lưu.")
        
    except KeyboardInterrupt:
        print("\nĐã dừng chương trình theo yêu cầu.")
    except Exception as e:
        print(f"Lỗi: {str(e)}")

if __name__ == "__main__":
    main() 