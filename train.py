import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import pandas as pd
from PIL import Image

# Tự động nhận diện đang chạy trên Kaggle hay trên máy Local
KAGGLE_DATA_PATH = "/kaggle/input/datasets/puneet6060/intel-image-classification"
if os.path.exists(KAGGLE_DATA_PATH):
    print(">> Đang chạy trên môi trường KAGGLE GPU")
    BASE_DATA = KAGGLE_DATA_PATH
    OUTPUT_DIR = "/kaggle/working"   # Thư mục duy nhất được phép ghi/lưu file trên Kaggle
else:
    print(">> Đang chạy trên môi trường LOCAL")
    BASE_DATA = "intel_data"
    OUTPUT_DIR = "."


# Đảm bảo console Windows in được tiếng Việt UTF-8 không bị lỗi charmap
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ==============================================================================
# 1. CẤU HÌNH (CONFIGURATION)
'''
- Dùng torch, torch.nn (xây mạng), torch.optim (thuật toán tối ưu)
- Dùng torchvision.transforms (tiền xử lý ảnh) và torchvision.models (lấy mô hình có sẵn)
- Dùng PIL.Image và pandas để xử lý ảnh và xuất file nộp bài
'''
# ==============================================================================
# Directory of dataset
TRAIN_DIR = os.path.join(BASE_DATA, "seg_train", "seg_train")
TEST_DIR  = os.path.join(BASE_DATA, "seg_test", "seg_test")
PRED_DIR  = os.path.join(BASE_DATA, "seg_pred", "seg_pred")

# Lưu checkpoint và file nộp bài vào OUTPUT_DIR
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

BATCH_SIZE = 32          # Trên máy local CPU để 16; khi lên Kaggle/Colab GPU để 32 hoặc 64
EPOCHS = 15               # Chạy thử 2 epoch trên local; khi train thật để 10-15
LEARNING_RATE = 1e-4     # Chuẩn cho fine-tuning pretrained model
NUM_CLASSES = 6
IMAGE_SIZE = (224, 224)

# Tự động chọn GPU nếu có, ngược lại dùng CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f">> Thiết bị đang sử dụng: {DEVICE}")

# ==============================================================================
# 2. TIỀN XỬ LÝ & DATA AUGMENTATION (TRANSFORMS)
'''
- Tập Train: bắt buộc có Augmentation (lật ảnh, xoay nhẹ, đổi màu) để tránh OverFiting
- Tập Val/ Test: KHÔNG augment, chỉ đổi kích thước và chuẩn hóa (Normalize)
- Bộ số mean=[0.485, 0.456, 0.406] và 
        std=[0.229, 0.224, 0.225] là chuẩn quốc tế của ImageNet
'''
# ==============================================================================
# Chuẩn hóa theo thống kê của tập ImageNet (Bắt buộc khi dùng pretrained model)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),              # đưa (150x150) -> (224x224)
    transforms.RandomHorizontalFlip(p=0.5),     # Lật ngang ảnh
    transforms.RandomRotation(degrees=15),       # Xoay nhẹ ảnh +/- 15 độ
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Chỉnh sáng/ tương phản
    transforms.ToTensor(),                       # Chuyển ảnh PIL sang Tensor [0, 1]
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

test_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
])

# ==============================================================================
# 3. TẠO DATASET & DATALOADER
# ==============================================================================
print(">> Đang tải dữ liệu...")
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transforms)
test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"-> Số ảnh Train: {len(train_dataset)} ({len(train_loader)} batches)")
print(f"-> Số ảnh Test:   {len(test_dataset)} ({len(test_dataset)} batches)")
print(f"-> Danh sách nhãn: {train_dataset.classes}")

# ==============================================================================
# 4. KHỞI TẠO MÔ HÌNH (TRANSFER LEARNING VỚI RESNET-18)
# ==============================================================================
print(">> Đang khởi tạo mô hình ResNet-18...")
# 1. Tải ResNet-18 có sẵn trọng số
# models.resnet18(weights=models.ResNet18_Weights.DEFAULT) tải trọng số ImageNet
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Thay thế lớp Fully Connected (fc) cuối cùng cho bài toán 6 classes
# 2. Lấy số lượng đặc trưng đầu vào của lớp cuối (với ResNet-18 là 512)
in_features  = model.fc.in_features

# 3. Thay bằng lớp Linear mới có NUM_CLASSES (6) ngõ ra
model.fc = nn.Linear(in_features , NUM_CLASSES)

# 4. Chuyển mô hình sang GPU/CPU
model = model.to(DEVICE)

# ==============================================================================
# 5. LOSS FUNCTION, OPTIMIZER & SCHEDULER
'''
- Hàm mất mát (Loss): nn.CrossEntropyLoss() (chuẩn cho bài toán phân loại nhiều lớp).
- Bộ tối ưu (Optimizer): optim.AdamW (tốt hơn Adam thường, có chống overfitting qua weight_decay).
- Scheduler: CosineAnnealingLR (tự động hạ dần Learning Rate theo hình cos).
'''
# ==============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ==============================================================================
# 6. HUẤN LUYỆN (TRAINING & VALIDATION LOOP)
'''
Mỗi Epoch gồm 2 giai đoạn:
1. model.train(): Tính loss, lan truyền ngược (loss.backward()), cập nhật trọng số (optimizer.step()).
2. model.eval() + torch.no_grad(): Chỉ đo độ chính xác trên tập Val, lưu lại file best_model.pth khi đạt kỷ lục accuracy mới.
'''
# ==============================================================================
best_val_acc = 0.0

print("\n" + "="*50)
print("BẮT ĐẦU HUẤN LUYỆN")
print("="*50)

for epoch in range(EPOCHS):
    start_time = time.time()
    
    # ---------- PHASE 1: TRAIN ----------
    model.train()
    train_loss, train_correct, total_train = 0.0, 0, 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()           # Xóa gradient cũ
        outputs = model(images)         # Dự đoán labels
        loss = criterion(outputs, labels)   # Tính loss
        loss.backward()                 # Đạo hàm lan truyền ngược
        optimizer.step()                # Cập nhật trọng số
        
        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data).item()
        total_train += labels.size(0)
        
    epoch_train_loss = train_loss / total_train
    epoch_train_acc = (train_correct / total_train) * 100.0
    
    # ---------- PHASE 2: VALIDATE ----------
    model.eval()
    val_loss, val_correct, total_val = 0.0, 0, 0
    
    with torch.no_grad():               # Tắt tính gradient để tiết kiệm RAM/tăng tốc
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data).item()
            total_val += labels.size(0)
            
    epoch_val_loss = val_loss / total_val
    epoch_val_acc = (val_correct / total_val) * 100.0
    
    scheduler.step()                    # Giảm dần learning rate
    duration = time.time() - start_time
    
    print(f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] ({duration:.1f}s) | "
          f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.2f}% | "
          f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.2f}%")
    
    # Lưu checkpoint tốt nhất
    if epoch_val_acc > best_val_acc:
        best_val_acc = epoch_val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"   --> Đã lưu model tốt nhất (Val Acc: {best_val_acc:.2f}%) vào {MODEL_SAVE_PATH}")

print(f"\n>> Hoan tat huan luyen. Val Acc tot nhat: {best_val_acc:.2f}%\n")

# ==============================================================================
# 7. SUY LUẬN (INFERENCE TRÊN TẬP SEG_PRED) VÀ XUẤT SUBMISSION
# ==============================================================================
if os.path.exists(PRED_DIR) and len(os.listdir(PRED_DIR)) > 0:
    print(">> Bắt đầu dự đoán trên tập seg_pred...")
    
    # Load model tốt nhất
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    model.eval()
    
    pred_images = [f for f in os.listdir(PRED_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    results = []
    
    with torch.no_grad():
        for img_name in pred_images:
            img_path = os.path.join(PRED_DIR, img_name)
            img = Image.open(img_path).convert("RGB")
            tensor = test_transforms(img).unsqueeze(0).to(DEVICE)
            
            output = model(tensor)
            _, predicted_idx = torch.max(output, 1)
            predicted_class = train_dataset.classes[predicted_idx.item()]
            
            results.append({"image_name": img_name, "predicted_label": predicted_class})
            
    df_submission = pd.DataFrame(results)
    df_submission.to_csv(SUBMISSION_PATH, index=False)
    print(f">> Đã lưu kết quả dự đoán vào file {SUBMISSION_PATH} (Top 5 kết quả):")
    print(df_submission.head())
