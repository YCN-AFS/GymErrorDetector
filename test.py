import cv2
import mediapipe as mp
# Khởi tạo Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()
# Cấu hình màu
joint_color = (255, 255, 255)   # Khớp: trắng
bone_color = (0, 255, 0)        # Xương: xanh lá
landmark_style = mp_drawing.DrawingSpec(color=joint_color, thickness=2, circle_radius=3)
connection_style = mp_drawing.DrawingSpec(color=bone_color, thickness=2)
# Mở video
video_path = r"C:\Users\fox\Documents\Projects\Physical-therapy\train\good\6111769848940.mp4"  # Đổi tên file tại đây
cap = cv2.VideoCapture(video_path)
# Kiểm tra video
if not cap.isOpened():
    print("Không thể mở video.")
    exit()
# Ghi video đầu ra
output_path = "output_pose_bone_green_joint_white.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
# Xử lý từng frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_style,
            connection_drawing_spec=connection_style
        )
    out.write(frame)
    cv2.imshow("Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()