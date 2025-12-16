# -*- coding: utf-8 -*-
"""
Sign Language Recognition - Video-to-Text (Phrase Classification)
Nhận diện video VSL → từ/cụm từ tiếng Việt (mỗi phrase = 1 class)
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm

# ==================== CONFIGURATION ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALID_DIR = os.path.join(BASE_DIR, 'valid')
TEST_DIR = os.path.join(BASE_DIR, 'test')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RECOGNITION_CONFIG = {
    'device': 'cpu',
    'batch_size': 4,
    'epochs': 50,
    'learning_rate': 1e-3,
    
    # Multi-frame sampling
    'frame_sample_rate': 5,
    'max_frames': 8,
    'image_size': 128,
    
    # Model architecture
    'cnn_hidden_dims': [32, 64, 128, 256],
    'lstm_hidden_dim': 256,
    'lstm_layers': 2,
    'dropout': 0.5,
}

print("=" * 60)
print("🎓 SIGN LANGUAGE RECOGNITION - Phrase Classification")
print("=" * 60)
print(f"🔧 Model: CNN + LSTM → Classification (1 phrase = 1 class)")
print(f"🖥️  Device: {RECOGNITION_CONFIG['device'].upper()}")
print("=" * 60)


# ==================== DATASET ====================
class SignLanguagePhraseDataset(Dataset):
    """Dataset nhận diện cụm từ (mỗi phrase là 1 class)"""
    
    def __init__(self, data_dir, config=RECOGNITION_CONFIG):
        self.data_dir = data_dir
        self.config = config
        
        self.videos_dir = os.path.join(data_dir, 'vid')
        self.labels_dir = os.path.join(data_dir, 'labels')
        
        # Load video files
        self.video_files = [f for f in os.listdir(self.videos_dir)
                            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        # Load labels (mỗi phrase nguyên vẹn là 1 class)
        self.labels = {}
        all_phrases = set()
        
        for video_name in self.video_files:
            base_name = os.path.splitext(video_name)[0]
            label_file = os.path.join(self.labels_dir, f"{base_name}.txt")
            
            if os.path.exists(label_file):
                with open(label_file, 'r', encoding='utf-8') as f:
                    phrase = f.read().strip()  # "Buổi sáng", "Cảm ơn", "Tôi"
                    self.labels[video_name] = phrase
                    all_phrases.add(phrase)
        
        # Filter videos có label
        self.video_files = [v for v in self.video_files if v in self.labels]
        
        # Build phrase → ID mapping (mỗi phrase nguyên vẹn = 1 class)
        self.phrase2id = {phrase: idx for idx, phrase in enumerate(sorted(all_phrases))}
        self.id2phrase = {idx: phrase for phrase, idx in self.phrase2id.items()}
        self.num_classes = len(self.phrase2id)
        
        print(f"📊 Dataset: {len(self.video_files)} videos")
        print(f"📝 Number of classes (phrases): {self.num_classes}")
        print(f"📝 Sample phrases: {list(all_phrases)[:10]}")
        print(f"📝 Class mapping examples:")
        for phrase, idx in list(self.phrase2id.items())[:5]:
            print(f"   '{phrase}' → class {idx}")
    
    def __len__(self):
        return len(self.video_files)
    
    def extract_frames(self, video_path):
        """Trích xuất frames từ video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")
        
        sample_rate = self.config['frame_sample_rate']
        max_frames = self.config['max_frames']
        
        frames = []
        frame_count = 0
        
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                # Preprocess frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (self.config['image_size'], self.config['image_size']))
                frame = frame.astype(np.float32) / 255.0
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)  # (3, H, W)
                frames.append(frame_tensor)
            
            frame_count += 1
        
        cap.release()
        
        # Padding nếu thiếu frames
        while len(frames) < max_frames:
            if frames:
                frames.append(frames[-1].clone())
            else:
                frames.append(torch.zeros(3, self.config['image_size'], self.config['image_size']))
        
        frames_tensor = torch.stack(frames[:max_frames])  # (max_frames, 3, H, W)
        return frames_tensor
    
    def __getitem__(self, idx):
        video_name = self.video_files[idx]
        video_path = os.path.join(self.videos_dir, video_name)
        
        # Extract frames
        frames = self.extract_frames(video_path)  # (max_frames, 3, H, W)
        
        # Get label (toàn bộ phrase)
        phrase = self.labels[video_name]  # "Buổi sáng", "Cảm ơn", "Tôi"
        label_id = self.phrase2id[phrase]  # Chuyển phrase → class ID
        
        return {
            'frames': frames,
            'label_id': torch.tensor(label_id, dtype=torch.long),
            'label_text': phrase
        }


# ==================== MODEL ====================
class SignLanguageRecognitionModel(nn.Module):
    """
    Phrase Recognition Model:
    1. CNN per-frame feature extraction
    2. LSTM temporal modeling
    3. Classification head (1 phrase = 1 class)
    """
    
    def __init__(self, num_classes, config=RECOGNITION_CONFIG):
        super().__init__()
        self.config = config
        
        # CNN Feature Extractor (per frame)
        layers = []
        in_channels = 3
        for out_channels in config['cnn_hidden_dims']:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*layers)
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, config['image_size'], config['image_size'])
            cnn_out = self.cnn(dummy)
            self.cnn_output_dim = cnn_out.view(1, -1).size(1)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_dim,
            hidden_size=config['lstm_hidden_dim'],
            num_layers=config['lstm_layers'],
            batch_first=True,
            dropout=config['dropout'] if config['lstm_layers'] > 1 else 0,
            bidirectional=True
        )
        
        # Classification head
        lstm_output_dim = config['lstm_hidden_dim'] * 2  # bidirectional
        self.classifier = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(lstm_output_dim // 2, num_classes)
        )
    
    def forward(self, frames):
        """
        Args:
            frames: (B, T, C, H, W) - T = num_frames
        
        Returns:
            logits: (B, num_classes)
        """
        batch_size, num_frames, C, H, W = frames.shape
        
        # 1. Extract per-frame features with CNN
        frames_flat = frames.view(batch_size * num_frames, C, H, W)
        cnn_features = self.cnn(frames_flat)  # (B*T, cnn_output_dim)
        cnn_features = cnn_features.view(batch_size, num_frames, -1)  # (B, T, cnn_output_dim)
        
        # 2. Temporal modeling with LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)  # lstm_out: (B, T, hidden*2)
        
        # 3. Use last hidden state for classification
        # h_n shape: (num_layers*2, B, hidden) for bidirectional
        # Concatenate forward and backward last hidden states
        h_forward = h_n[-2, :, :]  # (B, hidden)
        h_backward = h_n[-1, :, :]  # (B, hidden)
        h_concat = torch.cat([h_forward, h_backward], dim=1)  # (B, hidden*2)
        
        # 4. Classification
        logits = self.classifier(h_concat)  # (B, num_classes)
        
        return logits


# ==================== TRAINING ====================
def train_recognition(
    train_dir=TRAIN_DIR,
    valid_dir=VALID_DIR,
    epochs=RECOGNITION_CONFIG['epochs'],
    batch_size=RECOGNITION_CONFIG['batch_size'],
    learning_rate=RECOGNITION_CONFIG['learning_rate'],
    save_dir=MODELS_DIR
):
    device = torch.device(RECOGNITION_CONFIG['device'])
    print(f"\n🔥 Training on: {device}")
    
    # Datasets
    print("\n📊 Loading datasets...")
    train_dataset = SignLanguagePhraseDataset(train_dir)
    num_classes = train_dataset.num_classes
    phrase2id = train_dataset.phrase2id
    id2phrase = train_dataset.id2phrase
    
    if os.path.exists(os.path.join(valid_dir, 'vid')):
        valid_dataset = SignLanguagePhraseDataset(valid_dir)
        # Ensure same vocabulary
        valid_dataset.phrase2id = phrase2id
        valid_dataset.id2phrase = id2phrase
        valid_dataset.num_classes = num_classes
    else:
        valid_dataset = train_dataset
        print("⚠️  No validation set, using train for validation")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    print(f"\n🏗️  Building Recognition model...")
    print(f"📊 Number of classes: {num_classes}")
    print(f"📝 Classes: {list(id2phrase.values())[:10]}...")
    
    model = SignLanguageRecognitionModel(num_classes=num_classes)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Trainable parameters: {total_params:,}")
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        # ===== TRAINING =====
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            frames = batch['frames'].to(device)  # (B, T, C, H, W)
            labels = batch['label_id'].to(device)  # (B,)
            
            optimizer.zero_grad()
            
            # Forward
            logits = model(frames)  # (B, num_classes)
            loss = criterion(logits, labels)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # ===== VALIDATION =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # For detailed error analysis
        predictions_detail = []
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validation"):
                frames = batch['frames'].to(device)
                labels = batch['label_id'].to(device)
                label_texts = batch['label_text']
                
                logits = model(frames)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
                
                # Store predictions for analysis
                for i in range(len(predictions)):
                    pred_phrase = id2phrase[predictions[i].item()]
                    true_phrase = label_texts[i]
                    predictions_detail.append({
                        'true': true_phrase,
                        'predicted': pred_phrase,
                        'correct': (pred_phrase == true_phrase)
                    })
        
        val_loss /= len(valid_loader)
        val_acc = val_correct / val_total
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} ({train_correct}/{train_total})")
        print(f"  Val Loss:   {val_loss:.4f}   | Val Acc:   {val_acc:.4f} ({val_correct}/{val_total})")
        
        # Show some validation predictions
        if epoch % 5 == 0:
            print(f"\n📝 Sample Validation Predictions:")
            for pred in predictions_detail[:5]:
                status = "✅" if pred['correct'] else "❌"
                print(f"  {status} True: '{pred['true']}' | Predicted: '{pred['predicted']}'")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(save_dir, 'best_phrase_recognition.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': RECOGNITION_CONFIG,
                'num_classes': num_classes,
                'phrase2id': phrase2id,
                'id2phrase': id2phrase,
                'epoch': epoch,
                'val_acc': val_acc,
                'val_loss': val_loss
            }, model_path)
            print(f"✅ Saved best model (Val Acc: {val_acc:.4f})")
    
    # Save history
    history_path = os.path.join(RESULTS_DIR, 'phrase_recognition_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETE!")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"{'='*60}")
    
    return model


# ==================== INFERENCE ====================
def recognize_video(model, video_path, id2phrase, config=RECOGNITION_CONFIG):
    """Nhận diện video → phrase (cụm từ)"""
    device = torch.device(config['device'])
    model = model.to(device)
    model.eval()
    
    # Extract frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {video_path}")
    
    sample_rate = config['frame_sample_rate']
    max_frames = config['max_frames']
    
    frames = []
    frame_count = 0
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (config['image_size'], config['image_size']))
            frame = frame.astype(np.float32) / 255.0
            frames.append(torch.from_numpy(frame).permute(2, 0, 1))
        
        frame_count += 1
    
    cap.release()
    
    # Padding
    while len(frames) < max_frames:
        if frames:
            frames.append(frames[-1].clone())
        else:
            frames.append(torch.zeros(3, config['image_size'], config['image_size']))
    
    frames_tensor = torch.stack(frames[:max_frames]).unsqueeze(0).to(device)  # (1, T, 3, H, W)
    
    # Recognition
    with torch.no_grad():
        logits = model(frames_tensor)  # (1, num_classes)
        probs = torch.softmax(logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_id].item()
        
        # Get top-3 predictions
        top3_probs, top3_ids = torch.topk(probs[0], min(3, len(id2phrase)))
        top3_results = [
            {'phrase': id2phrase[idx.item()], 'confidence': prob.item()}
            for idx, prob in zip(top3_ids, top3_probs)
        ]
    
    recognized_phrase = id2phrase[pred_id]
    
    return {
        'phrase': recognized_phrase,
        'confidence': confidence,
        'class_id': pred_id,
        'top3': top3_results
    }


# ==================== EVALUATION ====================
def evaluate_on_test(model_path, test_dir=TEST_DIR):
    """Đánh giá model trên test set"""
    device = torch.device(RECOGNITION_CONFIG['device'])
    
    # Load model
    if not os.path.exists(model_path):
        print(f"❌ Model không tồn tại: {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    num_classes = checkpoint['num_classes']
    id2phrase = checkpoint['id2phrase']
    phrase2id = checkpoint['phrase2id']
    
    model = SignLanguageRecognitionModel(num_classes=num_classes, config=config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test dataset
    if not os.path.exists(os.path.join(test_dir, 'vid')):
        print(f"❌ Test folder không tồn tại: {test_dir}")
        return
    
    test_dataset = SignLanguagePhraseDataset(test_dir, config=config)
    test_dataset.phrase2id = phrase2id
    test_dataset.id2phrase = id2phrase
    test_dataset.num_classes = num_classes
    
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # Evaluate
    correct = 0
    total = 0
    predictions = []
    
    # Per-class accuracy
    class_correct = {phrase: 0 for phrase in phrase2id.keys()}
    class_total = {phrase: 0 for phrase in phrase2id.keys()}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            frames = batch['frames'].to(device)
            labels = batch['label_id'].to(device)
            label_texts = batch['label_text']
            
            logits = model(frames)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Store predictions
            for i in range(len(preds)):
                pred_phrase = id2phrase[preds[i].item()]
                true_phrase = label_texts[i]
                is_correct = (pred_phrase == true_phrase)
                
                predictions.append({
                    'true_label': true_phrase,
                    'predicted_label': pred_phrase,
                    'correct': is_correct
                })
                
                # Update per-class stats
                class_total[true_phrase] += 1
                if is_correct:
                    class_correct[true_phrase] += 1
    
    accuracy = correct / total
    
    # Calculate per-class accuracy
    per_class_accuracy = {}
    for phrase in phrase2id.keys():
        if class_total[phrase] > 0:
            per_class_accuracy[phrase] = class_correct[phrase] / class_total[phrase]
        else:
            per_class_accuracy[phrase] = 0.0
    
    # Save results
    results = {
        'overall_accuracy': accuracy,
        'total_samples': total,
        'correct_predictions': correct,
        'per_class_accuracy': per_class_accuracy,
        'predictions': predictions
    }
    
    results_path = os.path.join(RESULTS_DIR, 'test_evaluation_phrase.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"📊 TEST EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"\n📊 Per-Class Accuracy:")
    for phrase, acc in sorted(per_class_accuracy.items(), key=lambda x: x[1], reverse=True):
        count = class_total[phrase]
        print(f"  '{phrase}': {acc:.4f} ({class_correct[phrase]}/{count})")
    print(f"\nResults saved: {results_path}")
    print(f"{'='*60}")
    
    return results


# ==================== MAIN ====================
if __name__ == "__main__":
    print("\nChọn chức năng:")
    print("1. Train Phrase Recognition model")
    print("2. Recognize video → phrase")
    print("3. Evaluate on test set")
    print("0. Thoát")
    
    choice = input("\nNhập lựa chọn: ").strip()
    
    if choice == "1":
        epochs = input(f"Epochs (default {RECOGNITION_CONFIG['epochs']}): ").strip()
        epochs = int(epochs) if epochs else RECOGNITION_CONFIG['epochs']
        model = train_recognition(epochs=epochs)
    
    elif choice == "2":
        model_path = os.path.join(MODELS_DIR, 'best_phrase_recognition.pt')
        if not os.path.exists(model_path):
            print("❌ Model chưa train! Chạy option 1.")
        else:
            checkpoint = torch.load(model_path, map_location='cpu')
            model = SignLanguageRecognitionModel(
                num_classes=checkpoint['num_classes'],
                config=checkpoint['config']
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            id2phrase = checkpoint['id2phrase']
            
            video_path = input("Đường dẫn video: ").strip()
            if os.path.exists(video_path):
                result = recognize_video(model, video_path, id2phrase)
                print(f"\n{'='*60}")
                print(f"✅ RECOGNITION RESULT:")
                print(f"{'='*60}")
                print(f"📝 Phrase: {result['phrase']}")
                print(f"🎯 Confidence: {result['confidence']:.2%}")
                print(f"\n📊 Top 3 Predictions:")
                for i, pred in enumerate(result['top3'], 1):
                    print(f"  {i}. '{pred['phrase']}' ({pred['confidence']:.2%})")
                print(f"{'='*60}")
            else:
                print("❌ File không tồn tại!")
    
    elif choice == "3":
        model_path = os.path.join(MODELS_DIR, 'best_phrase_recognition.pt')
        if not os.path.exists(model_path):
            print("❌ Model chưa train! Chạy option 1.")
        else:
            evaluate_on_test(model_path)
    
    elif choice == "0":
        print("👋 Tạm biệt!")