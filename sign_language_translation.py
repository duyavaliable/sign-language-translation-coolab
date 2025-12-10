# -*- coding: utf-8 -*-
"""
Sign Language Translation - YOLOv8 Segmentation
Optimized for Local Environment
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
print("🔄 Đang khởi tạo YOLO model...")
yolo_model = None  # Sẽ load khi cần thiết
current_model_path = None  # Track model đang dùng

def get_yolo_model(model_path='yolov8n-seg.pt'):
    """
    Lazy loading YOLO model với khả năng reload
    
    Args:
        model_path: Đường dẫn đến YOLO model (mặc định: yolov8n-seg.pt)
    
    Returns:
        model: YOLO model instance
    """
    global yolo_model, current_model_path
    
    # Reload nếu model_path khác
    if yolo_model is None or current_model_path != model_path:
        if not os.path.exists(model_path):
            print(f"⚠️ Warning: Model '{model_path}' không tồn tại.")
            print(f"   Ultralytics sẽ tự động tải pretrained model từ internet.")
        
        yolo_model = YOLO(model_path)
        current_model_path = model_path
        print(f"✓ Đã tải YOLOv8 Segmentation model: {model_path}")
    
    return yolo_model

# ==================== CHỨC NĂNG XỬ LÝ ẢNH ====================
def extract_hand_region(image, model_path='yolov8n-seg.pt'):
    """
    Trích xuất vùng tay từ ảnh sử dụng YOLO segmentation
    
    Args:
        image: Ảnh đầu vào (numpy array)
        model_path: Đường dẫn đến YOLO segmentation model
    
    Returns:
        hand_box: Vùng ảnh chứa bàn tay hoặc ảnh trống nếu không phát hiện
    """
    model = get_yolo_model(model_path)
    results = model(image, verbose=False)
    
    # Lấy boxes từ segmentation results
    if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # Chọn box lớn nhất (giả sử là tay)
        areas = [(x2-x1)*(y2-y1) for x1, y1, x2, y2 in boxes]
        idx = areas.index(max(areas))
        x1, y1, x2, y2 = boxes[idx].astype(int)
        hand_box = image[y1:y2, x1:x2]
        
        # Vẽ bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, "Hand Detected (YOLOv8-seg)", (x1, y1-10),
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
    Hỗ trợ cả cấu trúc flat (images trực tiếp) và per-class (subfolders)
    
    Args:
        data_dir: Đường dẫn đến thư mục dataset
        num_samples: Số lượng mẫu hiển thị cho mỗi class
    """
    if data_dir is None:
        data_dir = os.path.join(DATASET_DIR, 'train', 'images')
    
    if not os.path.exists(data_dir):
        print(f"❌ Thư mục {data_dir} không tồn tại.")
        return
    
    # Kiểm tra cấu trúc: flat hoặc per-class
    image_files = [f for f in os.listdir(data_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if image_files:
        # Cấu trúc flat: tất cả ảnh trong images/
        print(f"📁 Dataset structure: FLAT (tất cả ảnh trong {data_dir})")
        classes = ['all_images']
        class_images = {classes[0]: image_files}
    else:
        # Cấu trúc per-class: images/<class>/*.jpg
        classes = sorted([d for d in os.listdir(data_dir)
                        if os.path.isdir(os.path.join(data_dir, d))])
        
        if not classes:
            print(f"❌ Không tìm thấy ảnh hoặc class trong {data_dir}")
            return
        
        print(f"📁 Dataset structure: PER-CLASS ({len(classes)} classes)")
        class_images = {}
        for cls in classes:
            class_dir = os.path.join(data_dir, cls)
            imgs = [f for f in os.listdir(class_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            class_images[cls] = imgs
    
    # Hiển thị
    num_classes = len(classes)
    fig = plt.figure(figsize=(12, 2*num_classes))
    
    for i, class_name in enumerate(classes):
        images = class_images[class_name]
        
        if not images:
            continue
        
        samples = images[:num_samples] if len(images) > num_samples else images
        
        for j, image_name in enumerate(samples):
            idx = i * num_samples + j + 1
            ax = fig.add_subplot(num_classes, num_samples, idx)
            
            # Xác định đường dẫn
            if class_name == 'all_images':
                image_path = os.path.join(data_dir, image_name)
            else:
                image_path = os.path.join(data_dir, class_name, image_name)
            
            img = cv2.imread(image_path)
            
            if img is not None:
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                ax.set_title(f"{class_name}" if class_name != 'all_images' else image_name[:15])
                ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    print(f"✓ Hiển thị {num_classes} class với tối đa {num_samples} mẫu/class")

# ==================== TẠO FILE CẤU HÌNH ====================
def create_dataset_yaml(use_all: bool = False):
    """
    Tạo file dataset.yaml cho YOLO Segmentation training
    
    Args:
        use_all: Nếu True, dùng toàn bộ train data cho cả training và validation
    
    Returns:
        dataset_yaml_path: Đường dẫn đến file yaml đã tạo
    """
    os.makedirs(TRAIN_DIR, exist_ok=True)
    
    dataset_yaml_path = os.path.join(TRAIN_DIR, 'dataset.yaml')
    val_path = 'train/images' if use_all else 'val/images'
    
    dataset_yaml_content = f"""\
# YOLOv8 Segmentation Dataset Configuration
# Sign Language Translation - 22 ASL Letters

path: {DATASET_DIR}
train: train/images
val: {val_path}

# Number of classes
nc: 22

# Class names (ASL letters, excluding F and J)
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
    
    print(f"✓ Đã tạo dataset.yaml tại: {dataset_yaml_path}")
    print(f"  - Training path: train/images")
    print(f"  - Validation path: {val_path}")
    print(f"  - use_all={use_all}")
    
    return dataset_yaml_path

# ==================== TRAINING ====================
def train_model(epochs=50, batch=16, imgsz=640, model_name='yolov8n-seg.pt', use_all: bool = False):
    """
    Huấn luyện YOLO Segmentation model
    
    Args:
        epochs: Số epochs training
        batch: Batch size
        imgsz: Kích thước ảnh input
        model_name: Tên pretrained model (mặc định: yolov8n-seg.pt)
        use_all: Dùng toàn bộ data cho training
    
    Returns:
        model: YOLO model đã train
        results: Kết quả training
    """
    print("=" * 60)
    print("🚀 BẮT ĐẦU TRAINING YOLOv8 SEGMENTATION MODEL")
    print("=" * 60)
    
    # Tạo dataset.yaml
    dataset_yaml_path = create_dataset_yaml(use_all=use_all)
    
    # Load pretrained segmentation model
    model = YOLO(model_name)
    print(f"\n✓ Đã load pretrained model: {model_name}")
    print(f"  - Task: Segmentation")
    print(f"  - Architecture: YOLOv8")
    
    # Kiểm tra dataset.yaml
    if not os.path.exists(dataset_yaml_path):
        raise FileNotFoundError(f"❌ dataset.yaml không tìm thấy: {dataset_yaml_path}")
    
    # Kiểm tra dataset structure
    train_images = os.path.join(DATASET_DIR, 'train', 'images')
    train_labels = os.path.join(DATASET_DIR, 'train', 'labels')
    val_images = os.path.join(DATASET_DIR, 'val', 'images')
    
    if not os.path.exists(train_images):
        raise FileNotFoundError(f"❌ Không tìm thấy: {train_images}")
    if not os.path.exists(train_labels):
        raise FileNotFoundError(f"❌ Không tìm thấy: {train_labels}")
    
    # Cảnh báo nếu không có validation data
    if not use_all and not os.path.exists(val_images):
        print(f"\n⚠️  WARNING: Validation path '{val_images}' không tồn tại!")
        print(f"   Khuyến nghị: Đặt use_all=True hoặc chuẩn bị validation data.")
        response = input("   Tiếp tục training? (y/n): ")
        if response.lower() != 'y':
            print("❌ Training bị hủy.")
            return None, None
    
    # Đếm số ảnh và labels
    num_train_images = len([f for f in os.listdir(train_images) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    num_train_labels = len([f for f in os.listdir(train_labels) 
                           if f.endswith('.txt')])
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  - Training images: {num_train_images}")
    print(f"  - Training labels: {num_train_labels}")
    print(f"  - Classes: 22 (ASL letters)")
    
    print(f"\n⚙️  Training Configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch}")
    print(f"  - Image size: {imgsz}")
    print(f"  - Task: segment")
    
    # Train model
    print("\n" + "=" * 60)
    print("🏋️  STARTING TRAINING...")
    print("=" * 60 + "\n")
    
    results = model.train(
        data=dataset_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=MODELS_DIR,
        name='sign_language_model',
        task='segment',
        patience=10,  # Early stopping
        save=True,
        plots=True
    )
    
    print("\n" + "=" * 60)
    print("✅ TRAINING HOÀN TẤT!")
    print("=" * 60)
    print(f"📁 Model được lưu tại: {MODELS_DIR}/sign_language_model/weights/best.pt")
    
    return model, results

# ==================== DỰ ĐOÁN ====================
def predict_image(model, image_path, save_result=True):
    """
    Dự đoán trên một ảnh với YOLOv8 Segmentation
    
    Args:
        model: YOLO segmentation model đã train
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
    results = model(image, verbose=False)
    
    # Lấy boxes và masks
    boxes_arr = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0].boxes, 'xyxy') else np.array([])
    confs = results[0].boxes.conf.cpu().numpy() if hasattr(results[0].boxes, 'conf') else np.array([])
    cls_ids = results[0].boxes.cls.cpu().numpy().astype(int) if hasattr(results[0].boxes, 'cls') else np.array([])
    
    image_with_prediction = image.copy()
    
    if boxes_arr.size > 0:
        for i, box in enumerate(boxes_arr):
            x1, y1, x2, y2 = box.astype(int)
            confidence = float(confs[i]) if i < len(confs) else 0.0
            class_id = int(cls_ids[i]) if i < len(cls_ids) else -1
            predicted_sign = model.names[class_id] if (hasattr(model, 'names') and class_id in model.names) else f"Class {class_id}"
            
            # Vẽ bounding box
            color = (0, 255, 0)
            cv2.rectangle(image_with_prediction, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ text
            text = f"{predicted_sign}: {confidence:.2f}"
            cv2.putText(image_with_prediction, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        print(f"✓ Phát hiện {len(boxes_arr)} đối tượng")
    else:
        print("⚠️ Không phát hiện đối tượng nào")
        cv2.putText(image_with_prediction, "No objects detected", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Hiển thị kết quả (với error handling cho headless env)
    try:
        cv2.imshow("Prediction Result - YOLOv8-seg", image_with_prediction)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"ℹ️  GUI không khả dụng (headless environment): {e}")
        print("   Kết quả sẽ được lưu vào file thay vì hiển thị.")
    
    # Lưu kết quả
    if save_result:
        os.makedirs(MODELS_DIR, exist_ok=True)
        output_path = os.path.join(MODELS_DIR, 'prediction_result.jpg')
        cv2.imwrite(output_path, image_with_prediction)
        print(f"✓ Đã lưu kết quả tại: {output_path}")
    
    return image_with_prediction

# ==================== MAIN ====================
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 SIGN LANGUAGE TRANSLATION - YOLOv8 SEGMENTATION")
    print("=" * 60)
    print(f"📁 Base Directory: {BASE_DIR}")
    print(f"📁 Dataset Directory: {DATASET_DIR}")
    print(f"📁 Models Directory: {MODELS_DIR}")
    print(f"🔧 Model: YOLOv8 Segmentation (yolov8n-seg.pt)")
    print("=" * 60)
    
    # Menu lựa chọn
    print("\nChọn chức năng:")
    print("1. Tạo file dataset.yaml")
    print("2. Visualize dataset")
    print("3. Train model (YOLOv8-seg)")
    print("4. Predict trên ảnh")
    print("0. Thoát")
    
    choice = input("\nNhập lựa chọn của bạn: ").strip()
    
    if choice == "1":
        use_all = input("Dùng toàn bộ data cho training? (y/n, mặc định n): ").strip().lower() == 'y'
        create_dataset_yaml(use_all=use_all)
    
    elif choice == "2":
        visualize_dataset()
    
    elif choice == "3":
        epochs = int(input("Nhập số epochs (mặc định 50): ").strip() or 50)
        batch = int(input("Nhập batch size (mặc định 16): ").strip() or 16)
        use_all = input("Dùng toàn bộ data cho training? (y/n, mặc định n): ").strip().lower() == 'y'
        model, results = train_model(epochs=epochs, batch=batch, use_all=use_all)
    
    elif choice == "4":
        model_path = input("Nhập đường dẫn model (để trống để dùng best.pt): ").strip()
        if not model_path:
            model_path = os.path.join(MODELS_DIR, 'sign_language_model', 'weights', 'best.pt')
        
        image_path = input("Nhập đường dẫn ảnh: ").strip()
        
        if os.path.exists(model_path) and os.path.exists(image_path):
            model = YOLO(model_path)
            predict_image(model, image_path)
        else:
            if not os.path.exists(model_path):
                print(f"❌ Model không tồn tại: {model_path}")
            if not os.path.exists(image_path):
                print(f"❌ Ảnh không tồn tại: {image_path}")
    
    elif choice == "0":
        print("👋 Tạm biệt!")
    
    else:
        print("❌ Lựa chọn không hợp lệ!")