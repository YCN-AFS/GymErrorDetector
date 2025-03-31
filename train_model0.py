import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import mediapipe as mp
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.utils import class_weight
from sklearn.model_selection import KFold
from visualization import plot_training_history, plot_confusion_matrix, plot_roc_curve
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

# Định nghĩa mô hình LSTM với PyTorch
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

# Đọc và chuẩn bị dữ liệu
data_frame = pd.read_csv("video_poses.csv")
X = data_frame.drop("class", axis=1)
label_map = {'good': 1.0, 'bad': 0.0}
y = data_frame['class'].map(label_map)

if y.isna().any():
    print("Warning: NaN values found in labels")
    y = y.fillna(0.0)

X = X.values.astype('float32')
y = y.values.astype('float32')

# Chia dữ liệu thành train, validation và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

# Chuyển đổi sang tensor và tạo DataLoader
X_train = torch.FloatTensor(X_train).unsqueeze(1)
y_train = torch.FloatTensor(y_train)
X_val = torch.FloatTensor(X_val).unsqueeze(1)
y_val = torch.FloatTensor(y_val)
X_test = torch.FloatTensor(X_test).unsqueeze(1)
y_test = torch.FloatTensor(y_test)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Cập nhật input_size
input_size = 132  # 33 landmarks x 4 features (x, y, z, visibility)

# Khởi tạo model và optimizer
model = PoseLSTM(input_size)
class_weights = class_weight.compute_class_weight('balanced', 
                                                classes=np.unique(y_train.numpy()), 
                                                y=y_train.numpy())
class_weights = torch.FloatTensor(class_weights)
criterion = nn.BCELoss(weight=class_weights[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# Training với validation
num_epochs = 100
best_val_loss = float('inf')
patience = 10
no_improve = 0

# Thêm lists để lưu history
train_losses = []
val_losses = []
train_accs = []
val_accs = []

kf = KFold(n_splits=5, shuffle=True)
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

    train_dataset_fold = TensorDataset(X_train_fold, y_train_fold)
    val_dataset_fold = TensorDataset(X_val_fold, y_val_fold)

    train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)
    val_loader_fold = DataLoader(val_dataset_fold, batch_size=batch_size)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_true = []
        
        for batch_X, batch_y in train_loader_fold:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_preds.extend((outputs.squeeze() > 0.5).float().tolist())
            train_true.extend(batch_y.tolist())
        
        train_loss /= len(train_loader_fold)
        train_accuracy = sum(1 for x, y in zip(train_preds, train_true) if x == y) / len(train_true)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader_fold:
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()
                val_preds.extend((outputs.squeeze() > 0.5).float().tolist())
                val_true.extend(batch_y.tolist())
        
        val_loss /= len(val_loader_fold)
        val_accuracy = sum(1 for x, y in zip(val_preds, val_true) if x == y) / len(val_true)
        
        print(f'Fold [{fold+1}/5], Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Early stopping và model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered")
                break
        
        scheduler.step(val_loss)

        # Thêm loss và accuracy vào lists
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

# Load best model và đánh giá cuối cùng
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

with torch.no_grad():
    final_val_outputs = model(X_val)
    final_val_preds = (final_val_outputs.squeeze() > 0.5).float()
    final_val_accuracy = (final_val_preds == y_val).float().mean()
    
    print("\nKết quả đánh giá cuối cùng trên tập validation:")
    print(f"Accuracy: {final_val_accuracy:.4f}")
    print("\nChi tiết dự đoán trên tập validation:")
    print(f"Số lượng 'good': {(final_val_preds == 1).sum().item()}")
    print(f"Số lượng 'bad': {(final_val_preds == 0).sum().item()}")
    print(f"Tổng số mẫu validation: {len(final_val_preds)}")
    
    # Tính toán thêm các metrics
    precision = precision_score(y_val.numpy(), final_val_preds.numpy())
    recall = recall_score(y_val.numpy(), final_val_preds.numpy())
    f1 = f1_score(y_val.numpy(), final_val_preds.numpy())
    
    print("\nChi tiết metrics trên tập validation:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    final_test_outputs = model(X_test)
    final_test_preds = (final_test_outputs.squeeze() > 0.5).float()
    final_test_accuracy = (final_test_preds == y_test).float().mean()
    
    print("\nKết quả đánh giá cuối cùng trên tập test:")
    print(f"Accuracy: {final_test_accuracy:.4f}")
    print("\nChi tiết dự đoán trên tập test:")
    print(f"Số lượng 'good': {(final_test_preds == 1).sum().item()}")
    print(f"Số lượng 'bad': {(final_test_preds == 0).sum().item()}")
    print(f"Tổng số mẫu test: {len(final_test_preds)}")
    
# Sau khi training xong:
plot_training_history(train_losses, val_losses, train_accs, val_accs)
plot_confusion_matrix(y_val, final_val_preds)
plot_roc_curve(y_val, final_val_outputs.squeeze())

# Thêm phân tích thống kê về dataset
print("\nThống kê về dataset:")
print(f"Tổng số mẫu: {len(data_frame)}")
print(f"Số lượng mẫu 'good': {len(data_frame[data_frame['class'] == 'good'])}")
print(f"Số lượng mẫu 'bad': {len(data_frame[data_frame['class'] == 'bad'])}")
print("\nPhân phối các đặc trưng:")
print(data_frame.describe())
    