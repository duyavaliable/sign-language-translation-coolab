# -*- coding: utf-8 -*-
"""
Sign Language Translation - YOLOv8 Detection
Optimized for Local Environment
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless backend, không cần Tcl/Tk
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==================== CẤU HÌNH ĐƯỜNG DẪN ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'train')        # Thư mục train (images + labels)
VALID_DIR = os.path.join(BASE_DIR, 'valid')        # Thư mục validation
TEST_DIR = os.path.join(BASE_DIR, 'test')          # Thư mục test
MODELS_DIR = os.path.join(BASE_DIR, 'models')      # Lưu trained models
CONFIG_DIR = BASE_DIR                               # dataset.yaml sẽ lưu ở root

# ==================== KHỞI TẠO MODEL ====================
print("🔄 Đang khởi tạo YOLO model...")
yolo_model = None  # Sẽ load khi cần thiết
current_model_path = None  # Track model đang dùng

def get_yolo_model(model_path='yolov8n.pt'):
    """
    Lazy loading YOLO model với khả năng reload
    
    Args:
        model_path: Đường dẫn đến YOLO model (mặc định: yolov8n.pt - Detection)
    
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
        print(f"✓ Đã tải YOLOv8 Detection model: {model_path}")
    
    return yolo_model

# ==================== CHỨC NĂNG XỬ LÝ ẢNH ====================
def extract_hand_region(image, model_path='yolov8n.pt'):
    """
    Trích xuất vùng tay từ ảnh sử dụng YOLO detection
    
    Args:
        image: Ảnh đầu vào (numpy array)
        model_path: Đường dẫn đến YOLO detection model
    
    Returns:
        hand_box: Vùng ảnh chứa bàn tay hoặc ảnh trống nếu không phát hiện
    """
    model = get_yolo_model(model_path)
    results = model(image, verbose=False)
    
    # Lấy boxes từ detection results
    if hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # Chọn box lớn nhất (giả sử là tay)
        areas = [(x2-x1)*(y2-y1) for x1, y1, x2, y2 in boxes]
        idx = areas.index(max(areas))
        x1, y1, x2, y2 = boxes[idx].astype(int)
        hand_box = image[y1:y2, x1:x2]
        
        # Vẽ bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, "Hand Detected (YOLOv8)", (x1, y1-10),
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
def visualize_dataset(data_type='train', num_samples=3):
    """
    Hiển thị mẫu từ dataset
    
    Args:
        data_type: Loại dataset ('train', 'valid', hoặc 'test')
        num_samples: Số lượng mẫu hiển thị
    """
    # Xác định thư mục dựa trên data_type
    if data_type == 'train':
        data_dir = TRAIN_DIR
    elif data_type == 'valid':
        data_dir = VALID_DIR
    elif data_type == 'test':
        data_dir = TEST_DIR
    else:
        print(f"❌ data_type không hợp lệ: {data_type}. Chọn 'train', 'valid', hoặc 'test'.")
        return
    
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')
    
    if not os.path.exists(images_dir):
        print(f"❌ Thư mục {images_dir} không tồn tại.")
        return
    
    # Lấy danh sách ảnh
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"❌ Không tìm thấy ảnh trong {images_dir}")
        return
    
    print(f"📁 Dataset: {data_type.upper()}")
    print(f"📁 Images directory: {images_dir}")
    print(f"📁 Labels directory: {labels_dir}")
    print(f"📊 Total images: {len(image_files)}")
    
    # Chọn mẫu ngẫu nhiên
    import random
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Hiển thị
    fig = plt.figure(figsize=(15, 5))
    
    for i, image_name in enumerate(samples):
        ax = fig.add_subplot(1, len(samples), i + 1)
        
        image_path = os.path.join(images_dir, image_name)
        img = cv2.imread(image_path)
        
        if img is not None:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            ax.set_title(f"{image_name[:20]}...")
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    print(f"✓ Hiển thị {len(samples)} mẫu từ {data_type} dataset")

# ==================== TẠO FILE CẤU HÌNH ====================
def create_dataset_yaml():
    """
    Tạo file dataset.yaml cho YOLO Detection training
    
    Returns:
        dataset_yaml_path: Đường dẫn đến file yaml đã tạo
    """
    dataset_yaml_path = os.path.join(CONFIG_DIR, 'dataset.yaml')
    
    # Kiểm tra các thư mục tồn tại
    train_images = os.path.join(TRAIN_DIR, 'images')
    train_labels = os.path.join(TRAIN_DIR, 'labels')
    valid_images = os.path.join(VALID_DIR, 'images')
    valid_labels = os.path.join(VALID_DIR, 'labels')
    
    # Cảnh báo nếu thiếu thư mục
    if not os.path.exists(train_images):
        print(f"⚠️  WARNING: {train_images} không tồn tại!")
    if not os.path.exists(train_labels):
        print(f"⚠️  WARNING: {train_labels} không tồn tại!")
    if not os.path.exists(valid_images):
        print(f"⚠️  WARNING: {valid_images} không tồn tại!")
    if not os.path.exists(valid_labels):
        print(f"⚠️  WARNING: {valid_labels} không tồn tại!")
    
    dataset_yaml_content = f"""\
# YOLOv8 Detection Dataset Configuration
# Sign Language Translation - 22 ASL Letters

# Đường dẫn tuyệt đối đến root directory
path: {BASE_DIR}

# Đường dẫn tương đối từ path
train: train/images
val: valid/images

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
    print(f"  - Training path: train/images & train/labels")
    print(f"  - Validation path: valid/images & valid/labels")
    print(f"  - Task: Detection (bounding box)")
    
    return dataset_yaml_path

# ==================== TRAINING ====================
def train_model(epochs=50, batch=16, imgsz=640, model_name='yolov8n.pt'):
    """
    Huấn luyện YOLO Detection model
    
    Args:
        epochs: Số epochs training
        batch: Batch size
        imgsz: Kích thước ảnh input
        model_name: Tên pretrained model 
                   - 'yolov8n.pt' (nano - nhanh)
                   - 'yolov8s.pt' (small)
                   - 'yolov8m.pt' (medium)
                   - 'yolov8l.pt' (large)
                   - 'yolov8x.pt' (xlarge - chính xác nhất)
    
    Returns:
        model: YOLO model đã train
        results: Kết quả training
    """
    print("=" * 60)
    print(f"🚀 BẮT ĐẦU TRAINING YOLOv8 DETECTION MODEL")
    print("=" * 60)
    
    # Tạo dataset.yaml
    dataset_yaml_path = create_dataset_yaml()
    
    # Load pretrained model
    model = YOLO(model_name)
    print(f"\n✓ Đã load pretrained model: {model_name}")
    print(f"  - Task: Detection (Bounding Box)")
    print(f"  - Architecture: YOLOv8")
    
    # Kiểm tra dataset structure
    train_images = os.path.join(TRAIN_DIR, 'images')
    train_labels = os.path.join(TRAIN_DIR, 'labels')
    valid_images = os.path.join(VALID_DIR, 'images')
    valid_labels = os.path.join(VALID_DIR, 'labels')
    
    if not os.path.exists(train_images):
        raise FileNotFoundError(f"❌ Không tìm thấy: {train_images}")
    if not os.path.exists(train_labels):
        raise FileNotFoundError(f"❌ Không tìm thấy: {train_labels}")
    if not os.path.exists(valid_images):
        print(f"⚠️  WARNING: {valid_images} không tồn tại!")
        print("   Training sẽ dùng train data để validation.")
    
    # Đếm số ảnh và labels
    num_train_images = len([f for f in os.listdir(train_images) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    num_train_labels = len([f for f in os.listdir(train_labels) 
                           if f.endswith('.txt')])
    
    num_valid_images = 0
    if os.path.exists(valid_images):
        num_valid_images = len([f for f in os.listdir(valid_images) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  - Training images: {num_train_images}")
    print(f"  - Training labels: {num_train_labels}")
    print(f"  - Validation images: {num_valid_images}")
    print(f"  - Classes: 22 (ASL letters)")
    
    print(f"\n⚙️  Training Configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch}")
    print(f"  - Image size: {imgsz}")
    print(f"  - Task: detect (bounding box)")
    
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
        name='sign_language_detection',
        task='detect',
        patience=10,  # Early stopping
        save=True,
        plots=True
    )
    
    print("\n" + "=" * 60)
    print("✅ TRAINING HOÀN TẤT!")
    print("=" * 60)
    print(f"📁 Model được lưu tại: {MODELS_DIR}/sign_language_detection/weights/best.pt")
    
    return model, results

# ==================== ĐÁNH GIÁ MODEL ====================
def evaluate_model(model_path, data_type='test'):
    """
    Đánh giá model trên test dataset
    
    Args:
        model_path: Đường dẫn đến trained model
        data_type: Loại dataset để đánh giá ('test' hoặc 'val')
    
    Returns:
        metrics: Kết quả đánh giá
    """
    print("=" * 60)
    print(f"📊 ĐÁNH GIÁ MODEL TRÊN {data_type.upper()} DATASET")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ Model không tồn tại: {model_path}")
        return None
    
    # Load model
    model = YOLO(model_path)
    print(f"✓ Đã load model: {model_path}")
    
    # Tạo dataset.yaml nếu chưa có
    dataset_yaml_path = os.path.join(CONFIG_DIR, 'dataset.yaml')
    if not os.path.exists(dataset_yaml_path):
        print("⚠️  dataset.yaml chưa tồn tại. Đang tạo...")
        create_dataset_yaml()
    
    # Validate model
    print(f"\n🔍 Đang đánh giá trên {data_type} dataset...\n")
    
    metrics = model.val(
        data=dataset_yaml_path,
        split=data_type,  # 'test' hoặc 'val'
        save_json=True,
        plots=True
    )
    
    print("\n" + "=" * 60)
    print("✅ ĐÁNH GIÁ HOÀN TẤT!")
    print("=" * 60)
    
    return metrics

# ==================== DỰ ĐOÁN ====================
def predict_image(model, image_path, save_result=True):
    """
    Dự đoán trên một ảnh với YOLOv8 Detection
    
    Args:
        model: YOLO detection model đã train
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
    
    # Lấy boxes
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
        cv2.imshow("Prediction Result - YOLOv8 Detection", image_with_prediction)
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
    print("🤖 SIGN LANGUAGE TRANSLATION - YOLOv8 DETECTION")
    print("=" * 60)
    print(f"📁 Base Directory: {BASE_DIR}")
    print(f"📁 Train Directory: {TRAIN_DIR}")
    print(f"📁 Valid Directory: {VALID_DIR}")
    print(f"📁 Test Directory: {TEST_DIR}")
    print(f"📁 Models Directory: {MODELS_DIR}")
    print(f"🔧 Model: YOLOv8 Detection (yolov8n.pt)")
    print("=" * 60)
    
    # Menu lựa chọn
    print("\nChọn chức năng:")
    print("1. Tạo file dataset.yaml")
    print("2. Visualize dataset")
    print("3. Train model (YOLOv8 Detection)")
    print("4. Evaluate model (Test dataset)")
    print("5. Predict trên ảnh")
    print("0. Thoát")
    
    choice = input("\nNhập lựa chọn của bạn: ").strip()
    
    if choice == "1":
        create_dataset_yaml()
    
    elif choice == "2":
        print("\nChọn dataset:")
        print("1. Train")
        print("2. Valid")
        print("3. Test")
        data_choice = input("Nhập lựa chọn (mặc định 1): ").strip() or "1"
        
        data_type_map = {"1": "train", "2": "valid", "3": "test"}
        data_type = data_type_map.get(data_choice, "train")
        
        visualize_dataset(data_type=data_type)
    
    elif choice == "3":
        print("\nChọn model size:")
        print("1. YOLOv8n (nano - nhanh nhất, ít chính xác)")
        print("2. YOLOv8s (small - cân bằng)")
        print("3. YOLOv8m (medium - chính xác hơn)")
        model_choice = input("Nhập lựa chọn (mặc định 1): ").strip() or "1"
        
        model_map = {
            "1": "yolov8n.pt",
            "2": "yolov8s.pt",
            "3": "yolov8m.pt"
        }
        model_name = model_map.get(model_choice, "yolov8n.pt")
        
        epochs = int(input("Nhập số epochs (mặc định 50): ").strip() or 50)
        batch = int(input("Nhập batch size (mặc định 16): ").strip() or 16)
        
        model, results = train_model(epochs=epochs, batch=batch, model_name=model_name)
    
    elif choice == "4":
        model_path = input("Nhập đường dẫn model (để trống để dùng best.pt): ").strip()
        if not model_path:
            model_path = os.path.join(MODELS_DIR, 'sign_language_detection', 'weights', 'best.pt')
        
        print("\nChọn dataset để đánh giá:")
        print("1. Test")
        print("2. Valid")
        eval_choice = input("Nhập lựa chọn (mặc định 1): ").strip() or "1"
        
        data_type = "test" if eval_choice == "1" else "val"
        
        evaluate_model(model_path, data_type=data_type)
    
    elif choice == "5":
        model_path = input("Nhập đường dẫn model (để trống để dùng best.pt): ").strip()
        if not model_path:
            model_path = os.path.join(MODELS_DIR, 'sign_language_detection', 'weights', 'best.pt')
        
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