# -*- coding: utf-8 -*-
"""
Sign Language Recognition - Video-to-Text (Phrase Classification)
Nhận diện video VSL → từ/cụm từ tiếng Việt (mỗi phrase = 1 class)
OPTIMIZED FOR SMALL DATASET (3-5 classes)
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
import random

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
    'batch_size': 8,           
    'epochs': 50,             
    'learning_rate': 5e-4,     
    
    # ✅ BALANCED FRAME SAMPLING (giữ chi tiết tay)
    'frame_sample_rate': 2,    
    'max_frames': 32,          
    'image_size': 96,         
    
    # ✅ SMALLER MODEL (phù hợp 3-5 classes)
    'cnn_hidden_dims': [16, 32, 64],  
    'lstm_hidden_dim': 128,           
    'lstm_layers': 1,                 
    'dropout': 0.4,                   # 
    
    # ✅ DATA AUGMENTATION
    'use_augmentation': True,
    'augment_prob': 0.5,
    
    # ✅ CLASS FILTERING (chỉ giữ 3-5 classes phổ biến nhất)
    'max_classes': 5,          # Giới hạn số classes
    'min_samples_per_class': 5, # Tối thiểu 5 samples/class
}

print("=" * 60)
print("🎓 SIGN LANGUAGE RECOGNITION - Optimized for Small Dataset")
print("=" * 60)
print(f"🔧 Model: CNN + LSTM → Classification")
print(f"📊 Max Classes: {RECOGNITION_CONFIG['max_classes']}")
print(f"🖥️  Device: {RECOGNITION_CONFIG['device'].upper()}")
print(f"🎨 Augmentation: {'ON' if RECOGNITION_CONFIG['use_augmentation'] else 'OFF'}")
print("=" * 60)


# ==================== HELPER: BUILD SHARED VOCABULARY ====================
def build_shared_vocabulary(data_dirs, max_classes=None, min_samples=1):
    """
    Build shared vocabulary from multiple datasets
    Filter to keep only top N classes with enough samples
    """
    all_phrases = {}  # phrase → count
    
    for data_dir in data_dirs:
        videos_dir = os.path.join(data_dir, 'vid')
        labels_dir = os.path.join(data_dir, 'labels')
        
        if not os.path.exists(videos_dir) or not os.path.exists(labels_dir):
            continue
        
        video_files = [f for f in os.listdir(videos_dir)
                       if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        for video_name in video_files:
            base_name = os.path.splitext(video_name)[0]
            label_file = os.path.join(labels_dir, f"{base_name}.txt")
            
            if os.path.exists(label_file):
                with open(label_file, 'r', encoding='utf-8') as f:
                    phrase = f.read().strip()
                    all_phrases[phrase] = all_phrases.get(phrase, 0) + 1
    
    print(f"\n📊 Found {len(all_phrases)} unique phrases across all datasets")
    print(f"📝 Phrase counts: {dict(sorted(all_phrases.items(), key=lambda x: x[1], reverse=True))}")
    
    # Filter by minimum samples
    filtered_phrases = {p: c for p, c in all_phrases.items() if c >= min_samples}
    print(f"✅ After filtering (min {min_samples} samples): {len(filtered_phrases)} phrases")
    
    # Keep only top N classes
    if max_classes and len(filtered_phrases) > max_classes:
        # Sort by count, keep top N
        sorted_phrases = sorted(filtered_phrases.items(), key=lambda x: x[1], reverse=True)
        top_phrases = dict(sorted_phrases[:max_classes])
        print(f"✅ Keeping top {max_classes} classes: {list(top_phrases.keys())}")
        filtered_phrases = top_phrases
    
    # Build mappings
    phrase2id = {phrase: idx for idx, phrase in enumerate(sorted(filtered_phrases.keys()))}
    id2phrase = {idx: phrase for phrase, idx in phrase2id.items()}
    
    print(f"\n📝 Final vocabulary ({len(phrase2id)} classes):")
    for phrase, idx in phrase2id.items():
        count = filtered_phrases[phrase]
        print(f"   '{phrase}' → class {idx} ({count} samples)")
    
    return phrase2id, id2phrase


# ==================== DATASET WITH AUGMENTATION ====================
class SignLanguagePhraseDataset(Dataset):
    """Dataset nhận diện cụm từ với augmentation"""
    
    def __init__(self, data_dir, config=RECOGNITION_CONFIG, phrase2id=None, id2phrase=None, is_training=False):
        self.data_dir = data_dir
        self.config = config
        self.is_training = is_training  # Enable augmentation only for training
        
        self.videos_dir = os.path.join(data_dir, 'vid')
        self.labels_dir = os.path.join(data_dir, 'labels')
        
        # Load video files
        self.video_files = [f for f in os.listdir(self.videos_dir)
                            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        # Load labels
        self.labels = {}
        for video_name in self.video_files:
            base_name = os.path.splitext(video_name)[0]
            label_file = os.path.join(self.labels_dir, f"{base_name}.txt")
            
            if os.path.exists(label_file):
                with open(label_file, 'r', encoding='utf-8') as f:
                    phrase = f.read().strip()
                    self.labels[video_name] = phrase
        
        # Use provided vocabulary or build new one
        if phrase2id is not None and id2phrase is not None:
            self.phrase2id = phrase2id
            self.id2phrase = id2phrase
        else:
            # Build from current dataset
            all_phrases = set(self.labels.values())
            self.phrase2id = {phrase: idx for idx, phrase in enumerate(sorted(all_phrases))}
            self.id2phrase = {idx: phrase for phrase, idx in self.phrase2id.items()}
        
        # Filter videos: only keep videos with labels in vocabulary
        self.video_files = [v for v in self.video_files 
                           if v in self.labels and self.labels[v] in self.phrase2id]
        
        self.num_classes = len(self.phrase2id)
        
        print(f"📊 Dataset ({data_dir}): {len(self.video_files)} videos, {self.num_classes} classes")
        
        # Count samples per class
        class_counts = {phrase: 0 for phrase in self.phrase2id.keys()}
        for video in self.video_files:
            phrase = self.labels[video]
            if phrase in class_counts:
                class_counts[phrase] += 1
        
        print(f"📊 Samples per class:")
        for phrase, count in sorted(class_counts.items()):
            print(f"   '{phrase}': {count} videos")
    
    def __len__(self):
        return len(self.video_files)
    
    def augment_frames(self, frames):
        """
        Apply random augmentation to frames
        Args:
            frames: (T, 3, H, W) tensor
        Returns:
            frames: augmented (T, 3, H, W) tensor
        """
        if not self.is_training or not self.config.get('use_augmentation', False):
            return frames
        
        augment_prob = self.config.get('augment_prob', 0.5)
        
        # 1. Random horizontal flip (mirror - useful for sign language)
        if random.random() < augment_prob:
            frames = torch.flip(frames, dims=[3])  # Flip width
        
        # 2. Random brightness adjustment (±20%)
        if random.random() < augment_prob:
            brightness_factor = 0.8 + random.random() * 0.4  # [0.8, 1.2]
            frames = frames * brightness_factor
            frames = torch.clamp(frames, 0, 1)
        
        # 3. Random contrast adjustment
        if random.random() < augment_prob:
            contrast_factor = 0.8 + random.random() * 0.4  # [0.8, 1.2]
            mean = frames.mean(dim=[2, 3], keepdim=True)
            frames = (frames - mean) * contrast_factor + mean
            frames = torch.clamp(frames, 0, 1)
        
        # 4. Random Gaussian noise (small)
        if random.random() < augment_prob:
            noise = torch.randn_like(frames) * 0.02
            frames = frames + noise
            frames = torch.clamp(frames, 0, 1)
        
        # 5. Random rotation (small angle ±5 degrees)
        if random.random() < augment_prob * 0.5:  # Lower probability
            angle = (random.random() - 0.5) * 10  # [-5, 5] degrees
            # Apply rotation to each frame
            frames_np = frames.permute(0, 2, 3, 1).numpy()  # (T, H, W, 3)
            rotated = []
            for frame in frames_np:
                center = (frame.shape[1] // 2, frame.shape[0] // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_frame = cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))
                rotated.append(rotated_frame)
            frames = torch.from_numpy(np.stack(rotated)).permute(0, 3, 1, 2).float()
        
        return frames
    
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
        
        # Apply augmentation (only if training)
        frames = self.augment_frames(frames)
        
        # Get label
        phrase = self.labels[video_name]
        label_id = self.phrase2id[phrase]
        
        return {
            'frames': frames,
            'label_id': torch.tensor(label_id, dtype=torch.long),
            'label_text': phrase
        }


# ==================== MODEL ====================
class SignLanguageRecognitionModel(nn.Module):
    """
    Optimized model for 3-5 classes
    """
    
    def __init__(self, num_classes, config=RECOGNITION_CONFIG):
        super().__init__()
        self.config = config
        
        # ✅ CNN with BatchNorm and Dropout
        layers = []
        in_channels = 3
        for i, out_channels in enumerate(config['cnn_hidden_dims']):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.2 if i < len(config['cnn_hidden_dims']) - 1 else 0.3),
            ])
            in_channels = out_channels
        
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.cnn = nn.Sequential(*layers)
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, config['image_size'], config['image_size'])
            cnn_out = self.cnn(dummy)
            self.cnn_output_dim = cnn_out.view(1, -1).size(1)
        
        # ✅ LSTM (unidirectional to reduce params)
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_dim,
            hidden_size=config['lstm_hidden_dim'],
            num_layers=config['lstm_layers'],
            batch_first=True,
            dropout=0 if config['lstm_layers'] == 1 else 0.3,
            bidirectional=False  # Tắt bidirectional (giảm 2x params)
        )
        
        # ✅ Simple classifier
        lstm_output_dim = config['lstm_hidden_dim']  # No *2 because unidirectional
        self.classifier = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(lstm_output_dim, num_classes)  # Direct to output
        )
    
    def forward(self, frames):
        """
        Args:
            frames: (B, T, C, H, W)
        Returns:
            logits: (B, num_classes)
        """
        batch_size, num_frames, C, H, W = frames.shape
        
        # CNN per-frame
        frames_flat = frames.view(batch_size * num_frames, C, H, W)
        cnn_features = self.cnn(frames_flat)  # (B*T, cnn_output_dim, 1, 1)
        cnn_features = cnn_features.view(batch_size, num_frames, -1)  # (B, T, cnn_output_dim)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)
        
        # Use last hidden state (unidirectional)
        h_last = h_n[-1, :, :]  # (B, hidden_dim)
        
        # Classify
        logits = self.classifier(h_last)
        
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
    
    # ✅ Build shared vocabulary (filter to top classes)
    print("\n📊 Building shared vocabulary...")
    data_dirs = [train_dir]
    if os.path.exists(os.path.join(valid_dir, 'vid')):
        data_dirs.append(valid_dir)
    
    phrase2id, id2phrase = build_shared_vocabulary(
        data_dirs,
        max_classes=RECOGNITION_CONFIG['max_classes'],
        min_samples=RECOGNITION_CONFIG['min_samples_per_class']
    )
    num_classes = len(phrase2id)
    
    if num_classes < 2:
        print(f"❌ ERROR: Only {num_classes} classes found. Need at least 2 classes!")
        print("💡 Try:")
        print("   - Reduce 'min_samples_per_class'")
        print("   - Add more training data")
        return None
    
    # Datasets
    print("\n📊 Loading datasets...")
    train_dataset = SignLanguagePhraseDataset(
        train_dir, 
        phrase2id=phrase2id, 
        id2phrase=id2phrase,
        is_training=True  # Enable augmentation
    )
    
    if os.path.exists(os.path.join(valid_dir, 'vid')):
        valid_dataset = SignLanguagePhraseDataset(
            valid_dir,
            phrase2id=phrase2id,
            id2phrase=id2phrase,
            is_training=False  # No augmentation for validation
        )
    else:
        valid_dataset = train_dataset
        print("⚠️  No validation set, using train for validation")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    print(f"\n🏗️  Building Recognition model...")
    print(f"📊 Number of classes: {num_classes}")
    print(f"📝 Classes: {list(id2phrase.values())}")
    
    model = SignLanguageRecognitionModel(num_classes=num_classes)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Trainable parameters: {total_params:,}")
    
    # ✅ Weighted loss (if class imbalance)
    class_counts = {}
    for video in train_dataset.video_files:
        phrase = train_dataset.labels[video]
        class_counts[phrase] = class_counts.get(phrase, 0) + 1
    
    weights = []
    for idx in range(num_classes):
        phrase = id2phrase[idx]
        count = class_counts.get(phrase, 1)
        weight = 1.0 / count
        weights.append(weight)
    
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    weights = weights / weights.sum() * num_classes  # Normalize
    
    print(f"📊 Class weights: {weights.tolist()}")
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    patience_counter = 0
    early_stop_patience = 20
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        # TRAINING
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            frames = batch['frames'].to(device)
            labels = batch['label_id'].to(device)
            
            optimizer.zero_grad()
            
            logits = model(frames)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # VALIDATION
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
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
        
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} ({train_correct}/{train_total})")
        print(f"  Val Loss:   {val_loss:.4f}   | Val Acc:   {val_acc:.4f} ({val_correct}/{val_total})")
        
        if epoch % 10 == 0 or epoch == epochs - 1:
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
            patience_counter = 0
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
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n⚠️ Early stopping after {epoch+1} epochs (no improvement for {early_stop_patience} epochs)")
                break
    
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
    """Nhận diện video → phrase"""
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
    
    frames_tensor = torch.stack(frames[:max_frames]).unsqueeze(0).to(device)
    
    # Recognition
    with torch.no_grad():
        logits = model(frames_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_id].item()
        
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
    
    if not os.path.exists(os.path.join(test_dir, 'vid')):
        print(f"❌ Test folder không tồn tại: {test_dir}")
        return
    
    test_dataset = SignLanguagePhraseDataset(test_dir, config=config, phrase2id=phrase2id, id2phrase=id2phrase, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    correct = 0
    total = 0
    predictions = []
    
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
            
            for i in range(len(preds)):
                pred_phrase = id2phrase[preds[i].item()]
                true_phrase = label_texts[i]
                is_correct = (pred_phrase == true_phrase)
                
                predictions.append({
                    'true_label': true_phrase,
                    'predicted_label': pred_phrase,
                    'correct': is_correct
                })
                
                class_total[true_phrase] += 1
                if is_correct:
                    class_correct[true_phrase] += 1
    
    accuracy = correct / total
    
    per_class_accuracy = {}
    for phrase in phrase2id.keys():
        if class_total[phrase] > 0:
            per_class_accuracy[phrase] = class_correct[phrase] / class_total[phrase]
        else:
            per_class_accuracy[phrase] = 0.0
    
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
    print("1. Train Phrase Recognition model (optimized for 3-5 classes)")
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