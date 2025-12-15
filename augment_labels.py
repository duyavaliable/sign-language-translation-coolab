# -*- coding: utf-8 -*-
"""
Augment Labels - Tự động nhân bản file labels
Tạo các file _0, _1, _2, ... _8 từ các file label hiện có
"""

import os
import shutil
import re
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_LABELS = os.path.join(BASE_DIR, "train", "labels")


def parse_label_filename(filename):
    """
    Parse tên file label để lấy base name và số thứ tự
    
    Args:
        filename: Tên file (vd: "toi_3.txt", "1_0.txt", "buoi_sang_2.txt")
    
    Returns:
        (base_name, number) hoặc None nếu không match pattern
        vd: ("toi", 3), ("1", 0), ("buoi_sang", 2)
    """
    # Pattern: base_name_number.txt
    match = re.match(r'^(.+)_(\d+)\.txt$', filename)
    if match:
        base_name = match.group(1)
        number = int(match.group(2))
        return base_name, number
    return None


def get_existing_labels(labels_dir):
    """
    Scan thư mục labels và nhóm các file theo base_name
    
    Returns:
        dict: {base_name: [list of existing numbers]}
        vd: {"toi": [0, 1, 2], "1": [0, 1], "buoi_sang": [0]}
    """
    if not os.path.exists(labels_dir):
        print(f"❌ Thư mục không tồn tại: {labels_dir}")
        return {}
    
    label_groups = defaultdict(list)
    
    for filename in os.listdir(labels_dir):
        if not filename.endswith('.txt'):
            continue
        
        parsed = parse_label_filename(filename)
        if parsed:
            base_name, number = parsed
            label_groups[base_name].append(number)
    
    # Sort numbers
    for base_name in label_groups:
        label_groups[base_name].sort()
    
    return dict(label_groups)


def augment_labels(labels_dir, max_number=8, dry_run=False):
    """
    Tạo các file label mới từ _0 đến max_number
    
    Args:
        labels_dir: Đường dẫn thư mục labels
        max_number: Số thứ tự tối đa (0-8 = 9 files)
        dry_run: Nếu True chỉ hiển thị preview, không tạo file
    """
    print("=" * 60)
    print("📝 AUGMENT LABELS - Nhân bản file labels")
    print("=" * 60)
    print(f"📁 Labels directory: {labels_dir}")
    print(f"🔢 Target: _0 đến _{max_number} (tổng {max_number + 1} files/label)")
    print(f"{'🔍 DRY RUN MODE - Không tạo file thật' if dry_run else '✅ EXECUTE MODE - Sẽ tạo file'}")
    print("=" * 60)
    
    # Get existing labels
    label_groups = get_existing_labels(labels_dir)
    
    if not label_groups:
        print("❌ Không tìm thấy file label nào!")
        return
    
    print(f"\n📊 Tìm thấy {len(label_groups)} base labels:")
    for base_name, numbers in sorted(label_groups.items()):
        print(f"  - {base_name}: {numbers}")
    
    # Process each base_name
    total_created = 0
    total_skipped = 0
    
    for base_name, existing_numbers in sorted(label_groups.items()):
        print(f"\n🔄 Processing: {base_name}")
        
        # Find a source file to copy content from
        source_number = existing_numbers[0]
        source_file = os.path.join(labels_dir, f"{base_name}_{source_number}.txt")
        
        # Read content
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"  📄 Source: {base_name}_{source_number}.txt")
        print(f"  📝 Content: {content.strip()}")
        
        # Create missing files
        for target_number in range(max_number + 1):
            if target_number in existing_numbers:
                print(f"    ⏭️  Skip {base_name}_{target_number}.txt (exists)")
                total_skipped += 1
                continue
            
            target_file = os.path.join(labels_dir, f"{base_name}_{target_number}.txt")
            
            if dry_run:
                print(f"    🔍 [DRY RUN] Would create: {base_name}_{target_number}.txt")
            else:
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"    ✅ Created: {base_name}_{target_number}.txt")
            
            total_created += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ AUGMENTATION COMPLETE!")
    print("=" * 60)
    print(f"📊 Summary:")
    print(f"  - Base labels: {len(label_groups)}")
    print(f"  - Files skipped (existing): {total_skipped}")
    print(f"  - Files {'would be ' if dry_run else ''}created: {total_created}")
    print(f"  - Total expected files: {len(label_groups) * (max_number + 1)}")
    
    if dry_run:
        print("\n💡 Chạy lại với --execute để tạo file thật")


def verify_labels(labels_dir, max_number=8):
    """
    Kiểm tra xem tất cả base_name đã có đủ _0 đến max_number chưa
    
    Args:
        labels_dir: Đường dẫn thư mục labels
        max_number: Số thứ tự tối đa
    """
    print("=" * 60)
    print("🔍 VERIFY LABELS")
    print("=" * 60)
    
    label_groups = get_existing_labels(labels_dir)
    
    if not label_groups:
        print("❌ Không tìm thấy file label nào!")
        return
    
    target_numbers = set(range(max_number + 1))
    complete = []
    incomplete = []
    
    for base_name, existing_numbers in sorted(label_groups.items()):
        existing_set = set(existing_numbers)
        missing = sorted(target_numbers - existing_set)
        
        if not missing:
            complete.append(base_name)
            print(f"✅ {base_name}: Complete ({len(existing_numbers)} files)")
        else:
            incomplete.append((base_name, missing))
            print(f"⚠️  {base_name}: Missing {missing}")
    
    print("\n" + "=" * 60)
    print(f"📊 Complete: {len(complete)}/{len(label_groups)}")
    print(f"⚠️  Incomplete: {len(incomplete)}/{len(label_groups)}")
    
    if incomplete:
        print("\n💡 Chạy augment_labels() để tạo file thiếu")


# ---- New: prune_labels ----
def prune_labels(labels_dir, max_number=2, dry_run=True):
    """
    Xóa các file label có index > max_number
    Args:
        labels_dir: thư mục labels
        max_number: giữ các file _0 .. _max_number
        dry_run: nếu True chỉ in ra, không xóa
    """
    print("=" * 60)
    print("🗑️  PRUNE LABELS - Xóa file vượt quá index")
    print("=" * 60)
    print(f"📁 Labels directory: {labels_dir}")
    print(f"🔢 Keep indices: 0 .. {max_number}")
    print(f"{'🔍 DRY RUN - Không xóa' if dry_run else '❗ EXECUTE - Sẽ xóa file'}")
    print("=" * 60)

    groups = get_existing_labels(labels_dir)
    if not groups:
        print("❌ Không tìm thấy file label nào.")
        return

    deleted = 0
    skipped = 0
    errors = []

    for base_name, nums in sorted(groups.items()):
        for n in sorted(nums):
            if n > max_number:
                fn = f"{base_name}_{n}.txt"
                path = os.path.join(labels_dir, fn)
                if dry_run:
                    print(f"  [DRY] Would remove: {fn}")
                    deleted += 1
                else:
                    try:
                        os.remove(path)
                        print(f"  Removed: {fn}")
                        deleted += 1
                    except Exception as e:
                        errors.append((fn, str(e)))

    print("\n" + "=" * 60)
    print("🧾 PRUNE SUMMARY")
    print("=" * 60)
    print(f"  - Files {'would be ' if dry_run else ''}deleted: {deleted}")
    if errors:
        print(f"  - Errors: {len(errors)}")
        for fn, err in errors[:10]:
            print(f"    - {fn}: {err}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Augment labels: Tự động nhân bản file labels từ _0 đến _N và/hoặc prune files > N"
    )
    parser.add_argument(
        "--max-number",
        type=int,
        default=8,
        help="Số thứ tự tối đa (mặc định: 8, tức _0 đến _8). Khi dùng --prune, chỉ giữ đến _max-number."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Thực hiện thay đổi (mặc định là dry-run)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Chỉ kiểm tra không tạo/xóa file"
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="Xóa các file label có index > --max-number (dry-run khi không có --execute)"
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=TRAIN_LABELS,
        help=f"Đường dẫn thư mục labels (mặc định: {TRAIN_LABELS})"
    )

    args = parser.parse_args()

    if args.verify:
        verify_labels(args.labels_dir, args.max_number)
    elif args.prune:
        prune_labels(args.labels_dir, max_number=args.max_number, dry_run=not args.execute)
    else:
        augment_labels(
            args.labels_dir,
            max_number=args.max_number,
            dry_run=not args.execute
        )