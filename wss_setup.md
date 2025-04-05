# 🔒 Hướng dẫn cài đặt WSS (WebSocket Secure)

## Giới thiệu

WebSocket Secure (WSS) là phiên bản bảo mật của WebSocket, sử dụng SSL/TLS để mã hóa kết nối giữa client và server. Việc triển khai WSS là cần thiết để:

- Bảo vệ dữ liệu người dùng
- Cho phép kết nối từ các trang web HTTPS
- Tuân thủ các tiêu chuẩn bảo mật

## Cài đặt SSL/TLS cho máy chủ

### 1. Tạo chứng chỉ SSL

#### Sử dụng OpenSSL tự tạo chứng chỉ (cho môi trường phát triển)

```bash
# Tạo thư mục chứa chứng chỉ
mkdir -p certs

# Tạo private key và chứng chỉ tự ký
openssl req -x509 -newkey rsa:4096 -keyout certs/key.pem -out certs/cert.pem -days 365 -nodes -subj "/CN=localhost"
```

#### Sử dụng Let's Encrypt (cho môi trường production)

```bash
# Cài đặt certbot
sudo apt-get update
sudo apt-get install certbot

# Lấy chứng chỉ (thay yourdomain.com bằng tên miền thực tế)
sudo certbot certonly --standalone -d yourdomain.com -d www.yourdomain.com
```

### 2. Cập nhật mã nguồn để sử dụng SSL

Chỉnh sửa file `exercise_api.py` để thêm cấu hình SSL:

```python
# Thêm vào cuối file
if __name__ == "__main__":
    import uvicorn
    
    # Cho môi trường phát triển
    # uvicorn.run(app, host="0.0.0.0", port=2222, ssl_keyfile="certs/key.pem", ssl_certfile="certs/cert.pem")
    
    # Cho môi trường production (Let's Encrypt)
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=2222, 
        ssl_keyfile="/etc/letsencrypt/live/yourdomain.com/privkey.pem", 
        ssl_certfile="/etc/letsencrypt/live/yourdomain.com/fullchain.pem"
    )
```

## Cấu hình Reverse Proxy (Nginx)

Sử dụng Nginx làm reverse proxy để quản lý SSL/TLS và cân bằng tải:

```nginx
server {
    listen 443 ssl;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    # SSL settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384;
    
    # WebSocket proxy
    location /squat {
        proxy_pass http://localhost:2222/squat;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /plank {
        proxy_pass http://localhost:2222/plank;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /pushup {
        proxy_pass http://localhost:2222/pushup;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /lunges {
        proxy_pass http://localhost:2222/lunges;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # API endpoints
    location / {
        proxy_pass http://localhost:2222;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$host$request_uri;
}
```

## Cập nhật client để sử dụng WSS

### HTML/JavaScript

```javascript
// Thay đổi WebSocket URL từ ws:// thành wss://
const socket = new WebSocket('wss://yourdomain.com/squat');

socket.onopen = function(event) {
    console.log('Connected to server');
};

socket.onmessage = function(event) {
    const data = JSON.parse(event.data);
    // Xử lý dữ liệu
};
```

### Mobile App

```dart
// Flutter example
final channel = WebSocketChannel.connect(
  Uri.parse('wss://yourdomain.com/squat'),
);

channel.stream.listen((message) {
  final data = jsonDecode(message);
  // Xử lý dữ liệu
});
```

## Kiểm tra kết nối WSS

Sử dụng công cụ sau để kiểm tra:

1. https://www.websocket.org/echo.html
2. Chrome DevTools (Network tab)
3. Wireshark để xác nhận traffic đã được mã hóa

## Khởi động máy chủ an toàn

```bash
# Cách 1: Trực tiếp
python exercise_api.py

# Cách 2: Sử dụng uvicorn với SSL
uvicorn exercise_api:app --host 0.0.0.0 --port 2222 --ssl-keyfile /etc/letsencrypt/live/yourdomain.com/privkey.pem --ssl-certfile /etc/letsencrypt/live/yourdomain.com/fullchain.pem

# Cách 3: Sử dụng systemd để chạy như service (khuyến nghị cho production)
sudo systemctl start exercise_api
```

## Khắc phục sự cố

1. **Lỗi chứng chỉ**: Kiểm tra đường dẫn và quyền truy cập chứng chỉ
2. **Lỗi kết nối**: Kiểm tra cổng 443 đã mở chưa
3. **Lỗi mixed content**: Đảm bảo tất cả kết nối đều sử dụng HTTPS 