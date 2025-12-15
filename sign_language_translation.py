# -*- coding: utf-8 -*-
"""
Sign Language Translation - VLM (BLIP-2 only)
Giữ lại: 1) Test 1 video (zero-shot BLIP-2)  3) Visualize results  5) Config frame sampling
"""
import os
import cv2
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# ==================== CẤU HÌNH ĐƯỜNG DẪN ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALID_DIR = os.path.join(BASE_DIR, 'valid')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== VLM CONFIGURATION ====================
VLM_CONFIG = {
    'model_type': 'blip2',  # fixed to BLIP-2
    'device': 'cuda' if os.environ.get('USE_CUDA') == '1' else 'cpu',
    'max_tokens': 100,
    'temperature': 0.3,
    'frame_sample_rate': 5,  # Lấy 1 frame mỗi N frames
    'max_frames': 10,  # Số frame tối đa để xử lý mỗi video
}

print("=" * 60)
print("🤖 SIGN LANGUAGE TRANSLATION - VLM (BLIP-2 ONLY)")
print("=" * 60)
print(f"📁 Base Directory: {BASE_DIR}")
print(f"📁 Train Directory: {TRAIN_DIR}")
print(f"📁 Valid Directory: {VALID_DIR}")
print(f"🔧 VLM Type: {VLM_CONFIG['model_type'].upper()}")
print(f"🖥️  Device: {VLM_CONFIG['device'].upper()}")
print(f"🎬 Frame Sample Rate: {VLM_CONFIG['frame_sample_rate']}")
print("=" * 60)

# ==================== VLM MODEL LOADER (BLIP-2 ONLY) ====================
vlm_model = None
vlm_processor = None

def load_blip2_model():
    """Load BLIP-2 model (Salesforce)"""
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    import torch

    print("\n🔄 Loading BLIP-2 model...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16 if VLM_CONFIG['device'] == 'cuda' else torch.float32
    )

    if VLM_CONFIG['device'] == 'cuda':
        model = model.to('cuda')

    print("✓ BLIP-2 model loaded successfully!")
    return model, processor

def get_vlm_model():
    """Lazy loading BLIP-2 only"""
    global vlm_model, vlm_processor
    if vlm_model is None:
        try:
            vlm_model, vlm_processor = load_blip2_model()
        except Exception as e:
            print(f"❌ Error loading BLIP-2: {e}")
            print("💡 Install: pip install transformers accelerate")
            raise
    return vlm_model, vlm_processor

# ==================== VIDEO PROCESSING ====================
def extract_frames_from_video(video_path, sample_rate=None, max_frames=None):
    if sample_rate is None:
        sample_rate = VLM_CONFIG['frame_sample_rate']
    if max_frames is None:
        max_frames = VLM_CONFIG['max_frames']
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frames = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 0)
    print(f"  📹 Video info: {total_frames} frames, {fps} FPS")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_rate == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
            if len(frames) >= max_frames:
                break
        frame_count += 1
    cap.release()
    print(f"  ✓ Extracted {len(frames)} frames")
    return frames

def get_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    middle_frame_idx = max(0, total_frames // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read frame from video: {video_path}")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

# ==================== PROMPT ====================
SIGN_LANGUAGE_PROMPT = """
You are an expert in Vietnamese Sign Language (VSL).
Look at the hand gesture in this image and identify which sign language word or phrase it represents.

Analyze the hand shape, position, and movement to determine the meaning.

Respond with the Vietnamese word or phrase that best matches the sign language gesture shown.
If you cannot determine the sign with confidence, respond with "UNKNOWN".

Sign:"""

def create_sign_language_prompt(custom_prompt=None):
    return custom_prompt if custom_prompt else SIGN_LANGUAGE_PROMPT

# ==================== PREDICTION ====================
def predict_with_vlm(image, prompt=None):
    model, processor = get_vlm_model()
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    prompt_text = create_sign_language_prompt(prompt)
    prediction = predict_with_transformer_vlm(model, processor, image, prompt_text)
    return prediction.strip(), 0.85

def predict_with_transformer_vlm(model, processor, image, prompt):
    import torch
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    if VLM_CONFIG['device'] == 'cuda':
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=VLM_CONFIG['max_tokens'],
            temperature=VLM_CONFIG['temperature'],
            do_sample=False
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def predict_video_multi_frame(video_path, use_all_frames=False):
    if use_all_frames:
        frames = extract_frames_from_video(video_path)
        predictions = []
        for i, frame in enumerate(frames):
            print(f"  Processing frame {i+1}/{len(frames)}...")
            pred, conf = predict_with_vlm(frame)
            predictions.append((pred, conf))
        pred_counts = {}
        for pred, conf in predictions:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        final_prediction = max(pred_counts, key=pred_counts.get)
        confidence = pred_counts[final_prediction] / len(predictions)
        return final_prediction, confidence, predictions
    else:
        frame = get_middle_frame(video_path)
        prediction, confidence = predict_with_vlm(frame)
        return prediction, confidence, [(prediction, confidence)]

# ==================== SINGLE VIDEO (save JSON) ====================
def predict_single_video(video_path, use_all_frames=False):
    print("=" * 60)
    print("🔍 SINGLE VIDEO PREDICTION (BLIP-2 zero-shot)")
    print("=" * 60)
    if not os.path.exists(video_path):
        print(f"❌ File không tồn tại: {video_path}")
        return None, 0.0
    print(f"📹 Video: {os.path.basename(video_path)}")
    try:
        prediction, confidence, frame_preds = predict_video_multi_frame(video_path, use_all_frames=use_all_frames)
        print(f"\n✅ Prediction: {prediction}")
        print(f"📊 Confidence: {confidence:.2%}")
        # save result JSON
        result = {
            'video': os.path.basename(video_path),
            'prediction': prediction,
            'confidence': confidence,
            'frame_predictions': [{'pred': p, 'conf': c} for p, c in frame_preds]
        }
        out = os.path.join(RESULTS_DIR, f'vlm_pred_{os.path.splitext(os.path.basename(video_path))[0]}.json')
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"💾 Result JSON saved to: {out}")
        # also save annotated middle frame image
        try:
            middle_frame = get_middle_frame(video_path)
            result_img = np.array(middle_frame)
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            text = f"VSL Sign: {prediction}"
            cv2.putText(result_img, text, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            img_out = os.path.join(RESULTS_DIR, f'prediction_{os.path.basename(video_path)}.jpg')
            cv2.imwrite(img_out, result_img)
            print(f"💾 Annotated image saved to: {img_out}")
        except Exception:
            pass
        return prediction, confidence
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None, 0.0

# ==================== VISUALIZE ====================
def visualize_video_predictions_from_file(result_file, num_display=1):
    """
    Hiển thị 1 result JSON file (saved by predict_single_video)
    """
    path = os.path.join(RESULTS_DIR, result_file)
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return
    with open(path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    video_name = result.get('video')
    video_path = os.path.join(VALID_DIR, 'vid', video_name)
    if not os.path.exists(video_path):
        video_path = os.path.join(TRAIN_DIR, 'vid', video_name)
    try:
        frame = get_middle_frame(video_path)
        plt.figure(figsize=(6,6))
        plt.imshow(frame)
        title = f"Pred: {result.get('prediction')} ({result.get('confidence'):.2%})"
        plt.title(title, color='green', fontweight='bold')
        plt.axis('off')
        out = os.path.join(RESULTS_DIR, f'viz_{os.path.splitext(result_file)[0]}.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Visualization saved to: {out}")
    except Exception as e:
        print(f"❌ Visualization error: {e}")

# ==================== CLI (options: 1,3,5,0) ====================
if __name__ == "__main__":
    print("\nChọn chức năng:")
    print("1. Test VLM trên một video (zero-shot BLIP-2)")
    print("3. Visualize prediction từ file kết quả (.json)")
    print("5. Config frame sampling")
    print("0. Thoát")
    choice = input("\nNhập lựa chọn: ").strip()
    if choice == "1":
        video_path = input("Nhập đường dẫn video: ").strip()
        use_all = input("Xử lý tất cả frames? (y/n, mặc định n): ").strip().lower() == 'y'
        predict_single_video(video_path, use_all_frames=use_all)
    elif choice == "3":
        result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json')]
        if not result_files:
            print("❌ Không có file kết quả nào!")
        else:
            print("\nFile kết quả có sẵn:")
            for i, f in enumerate(result_files):
                print(f"{i+1}. {f}")
            try:
                file_choice = int(input("Chọn file: ").strip()) - 1
                if 0 <= file_choice < len(result_files):
                    visualize_video_predictions_from_file(result_files[file_choice])
                else:
                    print("❌ Lựa chọn không hợp lệ")
            except Exception:
                print("❌ Lựa chọn không hợp lệ")
    elif choice == "5":
        try:
            rate = int(input(f"Frame sample rate (hiện tại: {VLM_CONFIG['frame_sample_rate']}): ").strip() or VLM_CONFIG['frame_sample_rate'])
            max_f = int(input(f"Max frames per video (hiện tại: {VLM_CONFIG['max_frames']}): ").strip() or VLM_CONFIG['max_frames'])
            VLM_CONFIG['frame_sample_rate'] = max(1, rate)
            VLM_CONFIG['max_frames'] = max(1, max_f)
            print(f"✓ Updated: Sample every {VLM_CONFIG['frame_sample_rate']} frames, max {VLM_CONFIG['max_frames']} frames per video")
        except Exception:
            print("❌ Input không hợp lệ")
    elif choice == "0":
        print("👋 Tạm biệt!")
    else:
        print("❌ Lựa chọn không hợp lệ!")