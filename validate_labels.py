import os

def validate_label_file(label_path):
    """
    Kiểm tra file label có đúng format DETECTION không (5 values: class xc yc w h)

    Returns:
        (is_valid, error_message)
    """
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        if not lines:
            return False, "File rỗng"

        for line_num, line in enumerate(lines, 1):
            parts = line.strip().split()

            # Detection format phải có đúng 5 giá trị
            if len(parts) != 5:
                return False, f"Dòng {line_num}: Phải có đúng 5 giá trị (class xc yc w h), nhưng có {len(parts)} giá trị"

            # Kiểm tra class_id là số nguyên
            try:
                class_id = int(parts[0])
                if class_id < 0 or class_id >= 22:
                    return False, f"Dòng {line_num}: class_id {class_id} không hợp lệ (phải 0-21)"
            except ValueError:
                return False, f"Dòng {line_num}: class_id '{parts[0]}' không phải số nguyên"

            # Kiểm tra bbox coords (xc, yc, w, h) là số thực trong khoảng [0, 1]
            coord_names = ['xc', 'yc', 'width', 'height']
            for i, (coord, name) in enumerate(zip(parts[1:5], coord_names), 1):
                try:
                    val = float(coord)
                    if val < 0 or val > 1:
                        return False, f"Dòng {line_num}: {name} = {val} ngoài khoảng [0, 1]"
                except ValueError:
                    return False, f"Dòng {line_num}: {name} '{coord}' không phải số thực"

        return True, "OK"

    except Exception as e:
        return False, f"Lỗi đọc file: {str(e)}"

def is_detection_line(parts):
    """Return True if parts correspond to YOLO detection line (5 values)."""
    if len(parts) != 5:
        return False
    # class id
    try:
        cls = int(parts[0])
        if cls < 0 or cls >= 22:
            return False
    except ValueError:
        return False
    # bbox coords (xc, yc, w, h) should be floats in [0,1]
    try:
        vals = list(map(float, parts[1:5]))
    except ValueError:
        return False
    for v in vals:
        if v < 0.0 or v > 1.0:
            return False
    return True

def is_detection_file(label_path):
    """Return True if every non-empty line in file is detection-format."""
    try:
        with open(label_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            return False
        for line in lines:
            parts = line.split()
            if not is_detection_line(parts):
                return False
        return True
    except Exception:
        return False

def list_non_detection_files(folder_path):
    """
    Scan folder và return list các file KHÔNG phải detection-format
    """
    if not os.path.exists(folder_path):
        print(f"❌ Folder không tồn tại: {folder_path}")
        return []
    txts = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    non_det = []
    for fn in sorted(txts):
        p = os.path.join(folder_path, fn)
        if not is_detection_file(p):
            non_det.append(fn)
    return non_det

def validate_dataset_folder(folder_path):
    """Validate tất cả labels trong folder (Detection format)"""
    if not os.path.exists(folder_path):
        print(f"❌ Folder không tồn tại: {folder_path}")
        return

    label_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    if not label_files:
        print(f"⚠️  Không có file .txt trong {folder_path}")
        return

    print(f"\n📂 Kiểm tra {folder_path}")
    print(f"📊 Tổng số files: {len(label_files)}")
    print("-" * 60)

    valid_count = 0
    invalid_files = []

    for filename in label_files:
        filepath = os.path.join(folder_path, filename)
        is_valid, error = validate_label_file(filepath)

        if is_valid:
            valid_count += 1
        else:
            invalid_files.append((filename, error))
            print(f"❌ {filename}")
            print(f"   Lỗi: {error}\n")

    print("-" * 60)
    print(f"✅ Valid (detection format): {valid_count}/{len(label_files)}")
    print(f"❌ Invalid (detection format): {len(invalid_files)}/{len(label_files)}")

    if invalid_files:
        print("\n⚠️  CÁC FILE LỖI:")
        for filename, error in invalid_files:
            print(f"  - {filename}: {error}")
    else:
        print("\n✅ Tất cả labels đều hợp lệ (detection format)!")

    return invalid_files

def check_label_format(folder):
    """Kiểm tra format của labels trong folder"""
    labels_dir = os.path.join(folder, 'labels')
    
    if not os.path.exists(labels_dir):
        print(f"❌ {labels_dir} không tồn tại")
        return
    
    files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    if not files:
        print(f"❌ Không có file .txt trong {labels_dir}")
        return
    
    # Kiểm tra 5 file đầu tiên
    sample_files = files[:5]
    
    print(f"\n📁 Folder: {folder}")
    print(f"📊 Tổng số file label: {len(files)}")
    print(f"\n📝 Mẫu label:")
    print("=" * 60)
    
    det_count = 0
    seg_count = 0
    other_count = 0
    
    for fn in sample_files:
        path = os.path.join(labels_dir, fn)
        with open(path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        
        if lines:
            first_line = lines[0]
            parts = first_line.split()
            
            print(f"\n{fn}:")
            print(f"  Số giá trị: {len(parts)}")
            print(f"  Nội dung: {first_line[:80]}...")
            
            if len(parts) == 5:
                print(f"  Định dạng: ✅ Detection (bbox)")
                det_count += 1
            elif len(parts) >= 7 and len(parts) % 2 == 1:
                print(f"  Định dạng: ⚠️  Segmentation (polygon) - CẦN CONVERT")
                seg_count += 1
            else:
                print(f"  Định dạng: ❌ Không xác định ({len(parts)} giá trị)")
                other_count += 1
    
    print("\n" + "=" * 60)
    print("📊 Tóm tắt định dạng:")
    print(f"  ✅ Detection (5 giá trị): {det_count}/{len(sample_files)}")
    print(f"  ⚠️  Segmentation (>5 giá trị): {seg_count}/{len(sample_files)}")
    print(f"  ❌ Không xác định: {other_count}/{len(sample_files)}")
    
    if seg_count > 0:
        print("\n⚠️  CẦN CHẠY CONVERT:")
        print("   python convert_seg_to_det.py --folders train valid test")
    
    print("=" * 60)

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 VALIDATE YOLO DETECTION LABELS")
    print("=" * 60)
    
    # Validate train labels
    print("\n1️⃣  TRAIN DATASET")
    train_invalid = validate_dataset_folder("train/labels")
    
    # Validate valid labels
    print("\n2️⃣  VALID DATASET")
    valid_invalid = validate_dataset_folder("valid/labels")
    
    # Validate test labels
    print("\n3️⃣  TEST DATASET")
    test_invalid = validate_dataset_folder("test/labels")
    
    # Check format summary
    folders = ['train', 'valid', 'test']
    for folder in folders:
        check_label_format(folder)
    
    print("\n" + "=" * 60)
    print("📊 TÓM TẮT CUỐI CÙNG")
    print("=" * 60)
    
    total_invalid = len(train_invalid or []) + len(valid_invalid or []) + len(test_invalid or [])
    
    if total_invalid == 0:
        print("✅ Tất cả labels đều hợp lệ (Detection format)!")
        print("   Dataset sẵn sàng để train YOLO Detection model.")
    else:
        print(f"⚠️  Tìm thấy {total_invalid} file labels lỗi hoặc chưa convert.")
        print("   Vui lòng:")
        print("   1. Chạy: python convert_seg_to_det.py --folders train valid test")
        print("   2. Kiểm tra lại: python validate_labels.py")