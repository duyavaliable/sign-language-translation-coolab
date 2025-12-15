import os
import random
import shutil
from typing import List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_VID = os.path.join(BASE_DIR, "train", "vid")
TRAIN_LABELS = os.path.join(BASE_DIR, "train", "labels")
VALID_VID = os.path.join(BASE_DIR, "valid", "vid")
VALID_LABELS = os.path.join(BASE_DIR, "valid", "labels")


def _list_videos(folder: str) -> List[str]:
    exts = (".mp4", ".avi", ".mov", ".mkv")
    if not os.path.isdir(folder):
        return []
    return [f for f in os.listdir(folder) if f.lower().endswith(exts) and os.path.isfile(os.path.join(folder, f))]


def split_train_valid(ratio: float = 0.8, move: bool = True, seed: int = 42):
    """
    Split train/vid + train/labels -> valid/vid + valid/labels
    Args:
        ratio: fraction to keep in train (e.g., 0.8)
        move: if True move files (train reduced). if False copy files (train kept).
        seed: random seed for reproducibility
    """
    os.makedirs(VALID_VID, exist_ok=True)
    os.makedirs(VALID_LABELS, exist_ok=True)

    videos = _list_videos(TRAIN_VID)
    if not videos:
        print(f"❌ No videos found in {TRAIN_VID}")
        return

    random.seed(seed)
    random.shuffle(videos)

    n_train = int(len(videos) * ratio)
    train_list = videos[:n_train]
    valid_list = videos[n_train:]

    print(f"Total videos: {len(videos)} -> Train: {len(train_list)} | Valid: {len(valid_list)}")
    moved_v, moved_l = 0, 0
    missing_labels = []

    for vid in valid_list:
        src_vid = os.path.join(TRAIN_VID, vid)
        dst_vid = os.path.join(VALID_VID, vid)

        base = os.path.splitext(vid)[0]
        label_name = f"{base}.txt"
        src_lbl = os.path.join(TRAIN_LABELS, label_name)
        dst_lbl = os.path.join(VALID_LABELS, label_name)

        try:
            if move:
                shutil.move(src_vid, dst_vid)
            else:
                shutil.copy2(src_vid, dst_vid)
            moved_v += 1
        except Exception as e:
            print(f"⚠️ Video error {vid}: {e}")

        if os.path.exists(src_lbl):
            try:
                if move:
                    shutil.move(src_lbl, dst_lbl)
                else:
                    shutil.copy2(src_lbl, dst_lbl)
                moved_l += 1
            except Exception as e:
                print(f"⚠️ Label move error {label_name}: {e}")
        else:
            missing_labels.append(label_name)

    print("\n=== RESULT ===")
    print(f"{'Moved' if move else 'Copied'} videos: {moved_v}")
    print(f"{'Moved' if move else 'Copied'} labels: {moved_l}")
    if missing_labels:
        print(f"Missing label files for {len(missing_labels)} videos (examples):")
        for p in missing_labels[:10]:
            print("  -", p)
    print(f"Train vids remaining: {len(_list_videos(TRAIN_VID))}")
    print(f"Valid vids now: {len(_list_videos(VALID_VID))}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split train videos/labels into train/valid (80/20 by default).")
    parser.add_argument("--ratio", type=float, default=0.8, help="Fraction to keep in train (0.0-1.0).")
    parser.add_argument("--move", action="store_true", help="Move files (default behavior); if not set, copies.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    # default: move files unless --move flag is omitted (to require explicit move, invert behavior)
    # Here we treat --move as signal to move; if not provided we copy for safety.
    split_train_valid(ratio=args.ratio, move=args.move, seed=args.seed)