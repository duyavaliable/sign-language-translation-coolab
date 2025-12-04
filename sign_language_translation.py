# -*- coding: utf-8 -*-
"""
Sign Language Translation - Optimized for Local Environment
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==================== CẤU HÌNH ĐƯỜNG DẪN ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TRAIN_DIR = os.path.join(BASE_DIR, 'train')

# ==================== KHỞI TẠO MODEL ====================
print("🔄 Đang tải YOLO model...")
yolo_model = None  # Sẽ load khi cần thiết

def get_yolo_model(model_path='yolov5su.pt'):
    """Lazy loading YOLO model"""
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLO(model_path)
        print(f"✓ Đã tải model: {model_path}")
    return yolo_model

# ==================== CHỨC NĂNG XỬ LÝ ẢNH ====================
def extract_hand_region(image, model_path='yolov5su.pt'):
    """
    Trích xuất vùng tay từ ảnh sử dụng YOLO detection
    
    Args:
        image: Ảnh đầu vào (numpy array)
        model_path: Đường dẫn đến YOLO model
    
    Returns:
        hand_box: Vùng ảnh chứa bàn tay hoặc ảnh trống nếu không phát hiện
    """
    model = get_yolo_model(model_path)
    results = model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    if len(boxes) > 0:
        # Chọn box lớn nhất (giả sử là tay)
        areas = [(x2-x1)*(y2-y1) for x1, y1, x2, y2 in boxes]
        idx = areas.index(max(areas))
        x1, y1, x2, y2 = boxes[idx].astype(int)
        hand_box = image[y1:y2, x1:x2]
        
        # Vẽ bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, "Hand Detected (YOLO)", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return hand_box
    
    # Không phát hiện tay
    cv2.putText(image, "No hand detected", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return np.zeros((64, 64, 3), dtype=np.uint8)

def draw_prediction(image, sign, confidence):
    """
    Vẽ kết quả dự đoán lên ảnh
    
    Args:
        image: Ảnh gốc
        sign: Ký hiệu được dự đoán
        confidence: Độ tin cậy
    
    Returns:
        result: Ảnh đã vẽ kết quả
    """
    result = image.copy()
    
    # Vẽ nền cho text
    overlay = result.copy()
    cv2.rectangle(overlay, (10, 10), (300, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)
    
    # Xác định text và màu
    if sign == "?" or confidence < 0.5:
        text = "Waiting for hand gesture..."
        color = (0, 0, 255)  # Red
    else:
        text = f"Sign: {sign} ({confidence:.2f})"
        color = (0, 255, 0)  # Green
    
    cv2.putText(result, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Vẽ khung hướng dẫn đặt tay
    height, width = image.shape[:2]
    roi_size = 300
    roi_x = width // 2 - roi_size // 2
    roi_y = height // 2 - roi_size // 2
    cv2.rectangle(result, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size),
                 (255, 0, 0), 2)
    
    return result

# ==================== HIỂN THỊ DATASET ====================
def visualize_dataset(data_dir=None, num_samples=3):
    """
    Hiển thị mẫu từ dataset
    
    Args:
        data_dir: Đường dẫn đến thư mục dataset
        num_samples: Số lượng mẫu hiển thị cho mỗi class
    """
    if data_dir is None:
        data_dir = os.path.join(DATASET_DIR, 'train', 'images')
    
    if not os.path.exists(data_dir):
        print(f"❌ Thư mục {data_dir} không tồn tại.")
        return
    
    # Lấy danh sách các class folders
    classes = sorted([d for d in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, d))])
    
    if not classes:
        print(f"❌ Không tìm thấy class nào trong {data_dir}")
        return
    
    num_classes = len(classes)
    fig = plt.figure(figsize=(12, 2*num_classes))
    
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        images = [f for f in os.listdir(class_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            continue
        
        samples = images[:num_samples] if len(images) > num_samples else images
        
        for j, image_name in enumerate(samples):
            idx = i * num_samples + j + 1
            ax = fig.add_subplot(num_classes, num_samples, idx)
            
            image_path = os.path.join(class_dir, image_name)
            img = cv2.imread(image_path)
            
            if img is not None:
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                ax.set_title(f"{class_name}")
                ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# ==================== TẠO FILE CẤU HÌNH ====================
def create_dataset_yaml(use_all: bool = False):
    """
    Tạo file dataset.yaml cho YOLO training
    If use_all True -> val points to train/images (use all data for training + val)
    """
    os.makedirs(TRAIN_DIR, exist_ok=True)
    
    dataset_yaml_path = os.path.join(TRAIN_DIR, 'dataset.yaml')
    val_path = 'train/images' if use_all else 'val/images'
    
    dataset_yaml_content = f"""\
path: {DATASET_DIR}
train: train/images
val: {val_path}

nc: 22
names:
  - A
  - B
  - C
  - D
  - E
  - G
  - H
  - I
  - K
  - L
  - M
  - N
  - O
  - P
  - Q
  - R
  - S
  - T
  - U
  - V
  - X
  - Y
"""
    
    with open(dataset_yaml_path, "w", encoding='utf-8') as f:
        f.write(dataset_yaml_content)
    print(f"✓ Đã tạo dataset.yaml tại: {dataset_yaml_path} (use_all={use_all})")
    
    return dataset_yaml_path

# ==================== TRAINING ====================
def train_model(epochs=50, batch=16, imgsz=640, model_name='yolov5n.pt', use_all: bool = False):
    """
    Huấn luyện YOLO model
    
    Args:
        epochs: Số epoch training
        batch: Batch size
        imgsz: Kích thước ảnh
        model_name: Tên pretrained model
    
    Returns:
        model: Model đã được train
        results: Kết quả training
    """
    print("🚀 Bắt đầu training...")
    
    # Tạo dataset.yaml
    dataset_yaml_path = create_dataset_yaml(use_all=use_all)
    
    # Kiểm tra dataset.yaml
    if not os.path.exists(dataset_yaml_path):
        raise FileNotFoundError(f"❌ dataset.yaml không tìm thấy: {dataset_yaml_path}")
    
    # Load model
    model = YOLO(model_name)
    print(f"✓ Đã load pretrained model: {model_name}")
    
    # Kiểm tra dataset structure
    train_images = os.path.join(DATASET_DIR, 'train', 'images')
    train_labels = os.path.join(DATASET_DIR, 'train', 'labels')
    val_images = os.path.join(DATASET_DIR, 'val', 'images')
    
    if not os.path.exists(train_images):
        raise FileNotFoundError(f"❌ Không tìm thấy: {train_images}")
    if not os.path.exists(train_labels):
        raise FileNotFoundError(f"❌ Không tìm thấy: {train_labels}")
    
    # Train
    print(f"📊 Training với {epochs} epochs, batch={batch}, imgsz={imgsz}")
    results = model.train(
        data=dataset_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=MODELS_DIR,
        name='sign_language_model'
    )
    
    print("✅ Training hoàn tất!")
    return model, results

# ==================== DỰ ĐOÁN ====================
def predict_image(model, image_path, save_result=True):
    """
    Dự đoán trên một ảnh
    
    Args:
        model: YOLO model đã train
        image_path: Đường dẫn đến ảnh
        save_result: Lưu kết quả hay không
    
    Returns:
        image_with_prediction: Ảnh với kết quả dự đoán
    """
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"❌ Lỗi: Không thể đọc ảnh từ {image_path}")
        return None
    
    # Dự đoán
    results = model(image)
    detections = results[0].boxes
    
    image_with_prediction = image.copy()
    
    if len(detections) > 0:
        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            predicted_sign = model.names[class_id] if hasattr(model, 'names') else f"Class {class_id}"
            
            # Vẽ bounding box
            color = (0, 255, 0)
            cv2.rectangle(image_with_prediction, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ text
            text = f"{predicted_sign}: {confidence:.2f}"
            cv2.putText(image_with_prediction, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        print(f"✓ Phát hiện {len(detections)} đối tượng")
    else:
        print("⚠️ Không phát hiện đối tượng nào")
        cv2.putText(image_with_prediction, "No objects detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Hiển thị kết quả
    cv2.imshow("Prediction Result", image_with_prediction)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Lưu kết quả
    if save_result:
        output_path = os.path.join(MODELS_DIR, 'prediction_result.jpg')
        cv2.imwrite(output_path, image_with_prediction)
        print(f"✓ Đã lưu kết quả tại: {output_path}")
    
    return image_with_prediction

# ==================== MAIN ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 SIGN LANGUAGE TRANSLATION - LOCAL VERSION")
    print("=" * 60)
    print(f"📁 Base Directory: {BASE_DIR}")
    print(f"📁 Dataset Directory: {DATASET_DIR}")
    print(f"📁 Models Directory: {MODELS_DIR}")
    print("=" * 60)
    
    # Menu lựa chọn
    print("\nChọn chức năng:")
    print("1. Tạo file dataset.yaml")
    print("2. Visualize dataset")
    print("3. Train model")
    print("4. Predict trên ảnh")
    print("0. Thoát")
    
    choice = input("\nNhập lựa chọn của bạn: ")
    
    if choice == "1":
        create_dataset_yaml()
    
    elif choice == "2":
        visualize_dataset()
    
    elif choice == "3":
        epochs = int(input("Nhập số epochs (mặc định 50): ") or 50)
        batch = int(input("Nhập batch size (mặc định 16): ") or 16)
        model, results = train_model(epochs=epochs, batch=batch)
    
    elif choice == "4":
        model_path = input("Nhập đường dẫn model (để trống để dùng best.pt): ").strip()
        if not model_path:
            model_path = os.path.join(MODELS_DIR, 'sign_language_model', 'weights', 'best.pt')
        
        image_path = input("Nhập đường dẫn ảnh: ").strip()
        
        if os.path.exists(model_path) and os.path.exists(image_path):
            model = YOLO(model_path)
            predict_image(model, image_path)
        else:
            print("❌ File không tồn tại!")
    
    elif choice == "0":
        print("👋 Tạm biệt!")
    
    else:
        print("❌ Lựa chọn không hợp lệ!")