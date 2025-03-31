# 🏋️‍♂️ Gym Exercise Form Evaluation System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.8+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

</div>

<p align="center">
Hệ thống AI đánh giá tư thế tập luyện thể dục thời gian thực, giúp người tập có thể tự điều chỉnh và cải thiện form tập một cách chính xác.
</p>

---

## 🌟 Tính năng chính

### 🎯 Phân tích chuyên sâu
- **Realtime Pose Detection** - Phân tích tư thế thời gian thực với MediaPipe
- **AI-Powered Evaluation** - Đánh giá chất lượng động tác bằng mô hình LSTM
- **High Performance** - Tối ưu hóa để chạy mượt mà trên CPU

### 💫 Giao diện thông minh
- **Dynamic UI** - Các box thông tin có thể kéo thả linh hoạt
- **Visual Feedback** - Phản hồi trực quan với màu sắc và biểu đồ
- **Detailed Analytics** - Phân tích chi tiết góc các khớp quan trọng

### 🔄 Phản hồi thời gian thực
- **Form Score** - Điểm số đánh giá chất lượng động tác (0-100%)
- **Smart Suggestions** - Gợi ý cải thiện thông minh
- **Performance Tracking** - Theo dõi tiến độ tập luyện

---

## 🛠️ Yêu cầu hệ thống

| Thư viện | Phiên bản | Mô tả |
|----------|-----------|--------|
| Python | ≥ 3.7 | Ngôn ngữ lập trình |
| OpenCV | ≥ 4.5 | Xử lý hình ảnh |
| PyTorch | ≥ 1.8 | Deep Learning framework |
| MediaPipe | ≥ 0.8 | Pose estimation |
| NumPy | ≥ 1.19 | Tính toán số học |
| Pillow | ≥ 8.0 | Xử lý hình ảnh phụ trợ |

## 📦 Cài đặt nhanh

```bash
# Clone repository
git clone <repository-url>
cd gym-form-evaluation

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy ứng dụng
python predict_realtime.py
```

## 🎮 Hướng dẫn sử dụng

### Khởi động
1. 📹 Chuẩn bị camera hoặc video
2. 🚀 Chạy chương trình
3. ⚙️ Điều chỉnh vị trí các box thông tin theo ý muốn

### Điều khiển
- 🖱️ **Chuột**: Kéo thả các box thông tin
- ⌨️ **Phím Q**: Thoát chương trình
- 🔄 **Tự động**: Các thông số được cập nhật realtime

## 🧠 Kiến trúc AI

### Model Architecture
```
┌─────────────────┐
│   Input Layer   │ 132 features
├─────────────────┤
│   LSTM Layer    │ hidden_size=128
├─────────────────┤
│  BatchNorm1d    │ 
│    Dropout      │ rate=0.5
├─────────────────┤
│  Linear Layers  │ 128 → 64 → 32 → 1
└─────────────────┘
```

### Performance Metrics
- **Accuracy**: ~95% trên tập test
- **Latency**: <50ms/frame trên CPU
- **FPS**: 20-30 fps trên hardware thông thường

## 📊 Giao diện trực quan

### Status Box 📈
```
┌──────────────────────┐
│ FPS: 30             │
│ Status: Good Form   │
│ [===========] 95%   │
└──────────────────────┘
```

### Feedback System 💭
```
┌──────────────────────┐
│ Exercise Feedback:   │
│ > Perfect posture   │
│ > Great balance     │
│ > Maintain form     │
└──────────────────────┘
```

## 🤝 Đóng góp

Chúng tôi luôn chào đón mọi đóng góp! Xem [CONTRIBUTING.md](CONTRIBUTING.md) để biết thêm chi tiết.

### Quy trình đóng góp
1. 🍴 Fork repository
2. 🌿 Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. 💾 Commit thay đổi (`git commit -m 'Add AmazingFeature'`)
4. 🚀 Push to branch (`git push origin feature/AmazingFeature`)
5. 🔍 Tạo Pull Request

## 📝 License

Copyright © 2024. Released under the [MIT License](LICENSE).

---

<div align="center">
Made with ❤️ for the fitness community
</div> 