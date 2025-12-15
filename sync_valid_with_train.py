import os
import shutil
from typing import Set
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_VID = os.path.join(BASE_DIR, "train", "vid")
TRAIN_LABELS = os.path.join(BASE_DIR, "train", "labels")
VALID_VID = os.path.join(BASE_DIR, "valid", "vid")
VALID_LABELS = os.path.join(BASE_DIR, "valid", "labels")


def _list_files(folder: str, exts: tuple) -> Set[str]:
    if not os.path.isdir(folder):
        return set()
    return {f for f in os.listdir(folder) if f.lower().endswith(exts) and os.path.isfile(os.path.join(folder, f))}


def sync_valid_from_train(move: bool = False, dry_run: bool = True):
    os.makedirs(VALID_VID, exist_ok=True)
    os.makedirs(VALID_LABELS, exist_ok=True)

    vid_exts = (".mp4", ".avi", ".mov", ".mkv")
    label_exts = (".txt",)

    train_vids = _list_files(TRAIN_VID, vid_exts)
    valid_vids = _list_files(VALID_VID, vid_exts)

    train_labels = _list_files(TRAIN_LABELS, label_exts)
    valid_labels = _list_files(VALID_LABELS, label_exts)

    # Videos missing in valid
    missing_vids = sorted(list(train_vids - valid_vids))
    # Labels missing in valid
    missing_labels = sorted(list(train_labels - valid_labels))

    print(f"Train vids: {len(train_vids)}  | Valid vids: {len(valid_vids)}  | Missing vids -> {len(missing_vids)}")
    print(f"Train labels: {len(train_labels)} | Valid labels: {len(valid_labels)} | Missing labels -> {len(missing_labels)}")

    copied_v, copied_l = 0, 0
    for vid in missing_vids:
        src = os.path.join(TRAIN_VID, vid)
        dst = os.path.join(VALID_VID, vid)
        if dry_run:
            print(f"[DRY] {'Move' if move else 'Copy'} video: {src} -> {dst}")
            copied_v += 1
        else:
            try:
                if move:
                    shutil.move(src, dst)
                else:
                    shutil.copy2(src, dst)
                print(f"{'Moved' if move else 'Copied'} video: {vid}")
                copied_v += 1
            except Exception as e:
                print(f"Error copying video {vid}: {e}")

        # also try to copy corresponding label (base_name.txt)
        base = os.path.splitext(vid)[0]
        label_name = f"{base}.txt"
        if label_name in train_labels and label_name not in valid_labels:
            src_lbl = os.path.join(TRAIN_LABELS, label_name)
            dst_lbl = os.path.join(VALID_LABELS, label_name)
            if dry_run:
                print(f"[DRY] {'Move' if move else 'Copy'} label: {src_lbl} -> {dst_lbl}")
                copied_l += 1
            else:
                try:
                    if move:
                        shutil.move(src_lbl, dst_lbl)
                    else:
                        shutil.copy2(src_lbl, dst_lbl)
                    print(f"{'Moved' if move else 'Copied'} label: {label_name}")
                    copied_l += 1
                except Exception as e:
                    print(f"Error copying label {label_name}: {e}")

    # Copy labels that are missing even if their videos already exist in valid
    for lbl in missing_labels:
        src = os.path.join(TRAIN_LABELS, lbl)
        dst = os.path.join(VALID_LABELS, lbl)
        # If label already copied above skip
        if lbl in train_labels and lbl not in valid_labels:
            if dry_run:
                print(f"[DRY] {'Move' if move else 'Copy'} label: {src} -> {dst}")
                copied_l += 1
            else:
                try:
                    if move:
                        shutil.move(src, dst)
                    else:
                        shutil.copy2(src, dst)
                    print(f"{'Moved' if move else 'Copied'} label: {lbl}")
                    copied_l += 1
                except Exception as e:
                    print(f"Error copying label {lbl}: {e}")

    print("\n=== SUMMARY ===")
    print(f"{'Would move' if dry_run else ('Moved' if move else 'Copied') } videos: {copied_v}")
    print(f"{'Would move' if dry_run else ('Moved' if move else 'Copied') } labels: {copied_l}")
    if dry_run:
        print("Dry run - no files were actually changed. Rerun with --no-dry to perform changes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync valid/vid and valid/labels from train if missing.")
    parser.add_argument("--move", action="store_true", help="Move files instead of copy.")
    parser.add_argument("--no-dry", action="store_true", help="Execute (default is dry-run).")
    args = parser.parse_args()

    sync_valid_from_train(move=args.move, dry_run=not args.no_dry)