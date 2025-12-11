import os
import argparse

def poly_line_to_det(parts):
    """
    Convert segmentation polygon line ['cls','x1','y1',... ] -> bbox (xc,yc,w,h)
    Supports arbitrary number of polygon points (even count).
    """
    cls = parts[0]
    coords = list(map(float, parts[1:]))
    
    # Kiểm tra số coords phải chẵn và >= 6 (ít nhất 3 điểm)
    if len(coords) < 6 or len(coords) % 2 != 0:
        raise ValueError(f"Invalid polygon: need at least 6 coords (3 points), got {len(coords)}")
    
    xs = coords[0::2]
    ys = coords[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    xc = (x_min + x_max) / 2.0
    yc = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return f"{cls} {xc:.12f} {yc:.12f} {w:.12f} {h:.12f}\n"

def is_detection_format(parts):
    """Kiểm tra xem line có phải detection format không (5 values)"""
    if len(parts) != 5:
        return False
    try:
        int(parts[0])  # class_id
        for val in parts[1:5]:
            float(val)  # coords
        return True
    except ValueError:
        return False

def is_segmentation_format(parts):
    """Kiểm tra xem line có phải segmentation format không (>5 values, số lẻ)"""
    # Segmentation: class_id + coords (số chẵn) = tổng số lẻ
    if len(parts) <= 5:
        return False
    if len(parts) % 2 != 1:  # Tổng phải lẻ (class + even coords)
        return False
    try:
        int(parts[0])  # class_id
        for val in parts[1:]:
            float(val)  # coords
        return True
    except ValueError:
        return False

def convert_folder_labels(root_folder, dry_run=False):
    """
    Convert tất cả label files trong folder từ segmentation → detection
    
    Args:
        root_folder: Đường dẫn đến folder (train/valid/test)
        dry_run: Nếu True, chỉ in ra không thay đổi file
    
    Returns:
        (số file converted, danh sách file paths)
    """
    labels_dir = os.path.join(root_folder, "labels")
    
    if not os.path.isdir(labels_dir):
        print(f"ℹ️  Skip (no labels): {labels_dir}")
        return 0, []
    
    converted = []
    skipped = []
    errors = []
    
    for fn in sorted(os.listdir(labels_dir)):
        if not fn.endswith(".txt"):
            continue
        
        path = os.path.join(labels_dir, fn)
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [L.strip() for L in f if L.strip()]
            
            if not lines:
                continue
            
            new_lines = []
            file_changed = False
            
            for line_num, line in enumerate(lines, 1):
                parts = line.split()
                
                if not parts:
                    continue
                
                # Đã là detection format → giữ nguyên
                if is_detection_format(parts):
                    new_lines.append(line + "\n")
                
                # Là segmentation format → convert
                elif is_segmentation_format(parts):
                    try:
                        new_lines.append(poly_line_to_det(parts))
                        file_changed = True
                    except ValueError as e:
                        errors.append((fn, line_num, str(e)))
                        new_lines.append(line + "\n")  # Giữ nguyên nếu lỗi
                
                # Format không xác định
                else:
                    errors.append((fn, line_num, f"Unknown format: {len(parts)} values"))
                    new_lines.append(line + "\n")
            
            if file_changed:
                converted.append(path)
                
                if not dry_run:
                    # Backup file gốc
                    backup_path = path + ".seg.bak"
                    if not os.path.exists(backup_path):  # Không ghi đè backup cũ
                        os.rename(path, backup_path)
                        
                        # Ghi file mới
                        with open(path, "w", encoding="utf-8") as f:
                            f.writelines(new_lines)
                    else:
                        # Nếu đã có backup, ghi đè file trực tiếp
                        with open(path, "w", encoding="utf-8") as f:
                            f.writelines(new_lines)
            else:
                skipped.append(fn)
        
        except Exception as e:
            errors.append((fn, 0, f"File error: {str(e)}"))
    
    print(f"✓ Converted: {len(converted)} files")
    print(f"ℹ️  Skipped (already detection): {len(skipped)} files")
    
    if errors:
        print(f"⚠️  Errors: {len(errors)}")
        # In 5 lỗi đầu tiên
        for fn, line, err in errors[:5]:
            if line > 0:
                print(f"   {fn}:{line} - {err}")
            else:
                print(f"   {fn} - {err}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more errors")
    
    return len(converted), converted

def main(folders, dry_run=False):
    """
    Chuyển đổi labels trong nhiều folders
    
    Args:
        folders: List các folder cần convert
        dry_run: Nếu True, chỉ in ra không thay đổi
    """
    print("=" * 60)
    print("🔄 CHUYỂN ĐỔI SEGMENTATION → DETECTION")
    print("=" * 60)
    
    total = 0
    all_converted = []
    
    for d in folders:
        print(f"\n📁 Processing folder: {d}")
        n, files = convert_folder_labels(d, dry_run=dry_run)
        total += n
        all_converted.extend(files)
    
    print("\n" + "=" * 60)
    print(f"📊 SUMMARY")
    print("=" * 60)
    print(f"Total files converted: {total}")
    
    if total and not dry_run:
        print("✓ Backup files saved as <file>.seg.bak")
        print("✓ Original files updated to detection format")
    
    if dry_run and total:
        print("ℹ️  Dry-run mode: no files were modified")
    
    if total == 0:
        print("ℹ️  All files are already in detection format or no segmentation labels found")
    
    return total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert YOLO segmentation labels (any polygon) → detection (bbox)"
    )
    parser.add_argument(
        "--folders", 
        nargs="+", 
        default=["train", "valid", "test"], 
        help="Folders to convert (default: train valid test)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Do not modify files; just report what would be changed"
    )
    
    args = parser.parse_args()
    main(args.folders, dry_run=args.dry_run)