# 📋 Tài liệu kỹ thuật - Gym Exercise Form Evaluation System

## 🎯 Tổng quan hệ thống

Hệ thống đánh giá tư thế tập luyện thể dục là một ứng dụng AI thời gian thực, kết hợp giữa computer vision và deep learning để phân tích và đánh giá chất lượng động tác của người tập.

### 🔄 Luồng xử lý chính

1. **Thu thập dữ liệu đầu vào**
   - Camera hoặc video stream
   - Frame rate: 30 FPS
   - Độ phân giải: 640x480

2. **Xử lý hình ảnh**
   - Chuyển đổi sang RGB
   - Resize và chuẩn hóa
   - Áp dụng MediaPipe Pose

3. **Phân tích tư thế**
   - Trích xuất 33 điểm mốc (landmarks)
   - Tính toán góc khớp
   - Chuẩn hóa dữ liệu

4. **Đánh giá chất lượng**
   - Mô hình LSTM phân tích chuỗi động tác
   - Tính điểm chất lượng (0-1)
   - Phân loại trạng thái

5. **Hiển thị kết quả**
   - Vẽ skeleton
   - Hiển thị góc khớp
   - Cập nhật UI realtime

## 🧠 Kiến trúc AI

### 1. Pose Detection (MediaPipe)

```python
# Cấu hình MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

- **Input**: Frame RGB
- **Output**: 33 landmarks (x, y, z, visibility)
- **Tốc độ**: ~30ms/frame

### 2. LSTM Model

```python
class PoseLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=132,  # 33 landmarks * 4 features
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
```

#### Thông số mô hình
- **Input size**: 132 features
  - 33 landmarks × 4 features (x, y, z, visibility)
- **Hidden size**: 128 units
- **Output size**: 1 (điểm chất lượng)
- **Dropout rate**: 0.5
- **Batch size**: 32
- **Learning rate**: 0.001

### 3. Feature Engineering

#### Trích xuất đặc trưng
1. **Landmark coordinates**
   - Normalize tọa độ (0-1)
   - Tính relative positions

2. **Joint angles**
   - Elbow angles (left/right)
   - Knee angles (left/right)
   - Hip angles
   - Shoulder angles

3. **Movement features**
   - Velocity
   - Acceleration
   - Range of motion

## 📊 Đánh giá hiệu năng

### 1. Accuracy Metrics
- **Training accuracy**: 98%
- **Validation accuracy**: 95%
- **Test accuracy**: 94%

### 2. Performance Metrics
- **FPS**: 20-30
- **Latency**: <50ms
- **Memory usage**: ~500MB
- **CPU usage**: 30-40%

### 3. Robustness
- **Lighting conditions**: Hoạt động tốt trong điều kiện ánh sáng vừa phải
- **Camera distance**: 1-3m
- **View angle**: 0-45 độ
- **Clothing**: Tối thiểu che khuất khớp

## 🔧 Tối ưu hóa

### 1. Performance Optimization
- **Frame skipping**: Bỏ qua frame khi FPS < 20
- **Batch processing**: Xử lý 32 frames/batch
- **Model quantization**: FP16
- **OpenMP parallelization**

### 2. Memory Management
- **Frame buffer**: 32 frames
- **Tensor recycling**
- **Garbage collection**

### 3. UI Optimization
- **Double buffering**
- **Lazy rendering**
- **Event-driven updates**

## 🛡️ Xử lý lỗi

### 1. Input Validation
```python
def validate_landmarks(landmarks):
    if not landmarks or len(landmarks) != 33:
        return False
    return all(0 <= l.x <= 1 and 0 <= l.y <= 1 for l in landmarks)
```

### 2. Error Recovery
- **Pose detection failure**: Sử dụng frame trước
- **Model inference error**: Reset model state
- **UI freeze**: Restart display thread

### 3. Logging
- **Error logs**: `error.log`
- **Performance logs**: `performance.log`
- **Debug logs**: `debug.log`

## 🔄 Quy trình huấn luyện

### 1. Data Collection
- **Dataset size**: 10,000 samples
- **Split ratio**: 70/15/15
- **Augmentation**:
  - Rotation (±15°)
  - Scaling (±10%)
  - Mirroring

### 2. Training Process
```python
# Training loop
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

### 3. Validation
- **Early stopping**: patience=10
- **Learning rate decay**: factor=0.1
- **Model checkpointing**

## 📈 Roadmap

### Phase 1: Core Features
- [x] Pose detection
- [x] Basic evaluation
- [x] Simple UI

### Phase 2: Enhancement
- [ ] Multi-person support
- [ ] Exercise recognition
- [ ] Custom exercise library

### Phase 3: Advanced Features
- [ ] Real-time coaching
- [ ] Progress tracking
- [ ] Social features

## 🔍 Troubleshooting

### Common Issues
1. **Low FPS**
   - Check CPU usage
   - Reduce frame resolution
   - Enable frame skipping

2. **Inaccurate Detection**
   - Adjust lighting
   - Check camera position
   - Update MediaPipe

3. **UI Lag**
   - Clear frame buffer
   - Reduce update frequency
   - Check memory usage

### Solutions
```python
# Performance monitoring
def monitor_performance():
    fps = calculate_fps()
    if fps < 20:
        enable_frame_skipping()
        reduce_resolution()
```

---

<div align="center">
*Last updated: March 2024*
</div> 