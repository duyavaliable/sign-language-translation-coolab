"""
Sign Language Training - DistilBERT + Multi-frame
Train model từ scratch với DistilBERT encoder + CNN vision encoder
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertModel,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW  
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

# ==================== CONFIGURATION ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALID_DIR = os.path.join(BASE_DIR, 'valid')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_CONFIG = {
    'device': 'cpu',  # CPU only
    'batch_size': 2,  # Nhỏ cho CPU
    'epochs': 20,
    'learning_rate': 2e-5,
    'max_length': 128,
    
    # Multi-frame config
    'frame_sample_rate': 10,  # Lấy 1 frame mỗi 10 frames
    'max_frames': 5,  # 5 frames/video (cố định)
    'image_size': 224,
    
    # Model architecture
    'text_encoder': 'distilbert-base-uncased',  # CPU-friendly
    'vision_hidden_dim': 128,
    'fusion_dim': 256,
}

print("=" * 60)
print("🎓 SIGN LANGUAGE TRAINING - DistilBERT + Multi-frame")
print("=" * 60)
print(f"📁 Train Directory: {TRAIN_DIR}")
print(f"📁 Valid Directory: {VALID_DIR}")
print(f"🔧 Model: DistilBERT + CNN")
print(f"🖥️  Device: {TRAIN_CONFIG['device'].upper()}")
print(f"🎬 Frames per video: {TRAIN_CONFIG['max_frames']} (fixed)")
print(f"📊 Batch Size: {TRAIN_CONFIG['batch_size']}")
print("=" * 60)

# ==================== DATASET (MULTI-FRAME) ====================
class MultiFrameSignLanguageDataset(Dataset):
    """Dataset với MULTI-FRAME support"""
    
    def __init__(self, data_dir, tokenizer, config=TRAIN_CONFIG):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.config = config
        
        self.videos_dir = os.path.join(data_dir, 'vid')
        self.labels_dir = os.path.join(data_dir, 'labels')
        
        # Load video files
        self.video_files = [f for f in os.listdir(self.videos_dir)
                            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        # Load labels
        self.labels = {}
        self.unique_labels = set()
        
        for video_name in self.video_files:
            base_name = os.path.splitext(video_name)[0]
            label_file = os.path.join(self.labels_dir, f"{base_name}.txt")
            
            if os.path.exists(label_file):
                with open(label_file, 'r', encoding='utf-8') as f:
                    label = f.read().strip()
                    self.labels[video_name] = label
                    self.unique_labels.add(label)
        
        # Filter videos có label
        self.video_files = [v for v in self.video_files if v in self.labels]
        
        # Create label mappings
        self.label2id = {label: idx for idx, label in enumerate(sorted(self.unique_labels))}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        print(f"📊 Dataset: {len(self.video_files)} videos, {len(self.unique_labels)} classes")
        print(f"📝 Classes: {list(self.label2id.keys())[:10]}...")
    
    def __len__(self):
        return len(self.video_files)
    
    def extract_multi_frames(self, video_path):
        """
        Trích xuất NHIỀU frames từ video (fixed number)
        
        Returns:
            frames: Tensor shape (num_frames, C, H, W)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)  # (C, H, W)
                frames.append(frame_tensor)
            
            frame_count += 1
        
        cap.release()
        
        # Padding nếu không đủ frames
        while len(frames) < max_frames:
            # Duplicate last frame
            frames.append(frames[-1] if frames else torch.zeros(3, self.config['image_size'], self.config['image_size']))
        
        # Stack frames: (num_frames, C, H, W)
        frames_tensor = torch.stack(frames[:max_frames])
        
        return frames_tensor
    
    def __getitem__(self, idx):
        video_name = self.video_files[idx]
        video_path = os.path.join(self.videos_dir, video_name)
        
        # Extract MULTI frames
        frames = self.extract_multi_frames(video_path)  # Shape: (5, 3, 224, 224)
        
        # Get label
        label_text = self.labels[video_name]
        label_id = self.label2id[label_text]
        
        # Tokenize label
        tokens = self.tokenizer(
            label_text,
            max_length=self.config['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'frames': frames,  # (num_frames, C, H, W)
            'label_id': torch.tensor(label_id, dtype=torch.long),
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'label_text': label_text
        }


# ==================== MODEL ARCHITECTURE (MULTI-FRAME) ====================
class MultiFrameSignLanguageModel(nn.Module):
    """
    Multi-frame model:
    1. Vision Encoder (CNN per frame)
    2. Temporal Aggregation (mean pooling)
    3. Text Encoder (DistilBERT)
    4. Fusion + Classifier
    """
    
    def __init__(self, num_classes, config=TRAIN_CONFIG):
        super().__init__()
        self.config = config
        
        # Vision Encoder (xử lý từng frame)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        
        # Text Encoder (DistilBERT)
        self.text_encoder = DistilBertModel.from_pretrained(config['text_encoder'])
        
        # Fusion layer
        vision_dim = config['vision_hidden_dim']
        text_dim = self.text_encoder.config.hidden_size  # 768 for DistilBERT
        
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + text_dim, config['fusion_dim']),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Classifier
        self.classifier = nn.Linear(config['fusion_dim'], num_classes)
    
    def forward(self, frames, input_ids, attention_mask):
        """
        Args:
            frames: (B, num_frames, C, H, W)
            input_ids: (B, seq_len)
            attention_mask: (B, seq_len)
        
        Returns:
            logits: (B, num_classes)
        """
        batch_size, num_frames, C, H, W = frames.shape
        
        # 1. Vision encoding (per frame)
        # Reshape: (B*num_frames, C, H, W)
        frames_flat = frames.view(batch_size * num_frames, C, H, W)
        
        # Encode: (B*num_frames, 128)
        vision_features_flat = self.vision_encoder(frames_flat)
        
        # Reshape back: (B, num_frames, 128)
        vision_features = vision_features_flat.view(batch_size, num_frames, -1)
        
        # 2. Temporal aggregation (mean pooling)
        vision_features_agg = vision_features.mean(dim=1)  # (B, 128)
        
        # 3. Text encoding
        text_output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_output.last_hidden_state[:, 0, :]  # (B, 768) CLS token
        
        # 4. Fusion
        combined = torch.cat([vision_features_agg, text_features], dim=1)  # (B, 896)
        fused = self.fusion(combined)  # (B, 256)
        
        # 5. Classify
        logits = self.classifier(fused)  # (B, num_classes)
        
        return logits


# ==================== TRAINING ====================
def train_model(
    train_dir=TRAIN_DIR,
    valid_dir=VALID_DIR,
    epochs=TRAIN_CONFIG['epochs'],
    batch_size=TRAIN_CONFIG['batch_size'],
    learning_rate=TRAIN_CONFIG['learning_rate'],
    save_dir=MODELS_DIR
):
    """
    Train multi-frame Sign Language model với DistilBERT
    """
    
    device = torch.device(TRAIN_CONFIG['device'])
    print(f"\n🔥 Training on: {device}")
    
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(TRAIN_CONFIG['text_encoder'])
    
    # Create datasets
    print("\n📊 Loading datasets...")
    train_dataset = MultiFrameSignLanguageDataset(train_dir, tokenizer)
    
    if os.path.exists(os.path.join(valid_dir, 'vid')):
        valid_dataset = MultiFrameSignLanguageDataset(valid_dir, tokenizer)
    else:
        print("⚠️  Valid dataset not found, using train for validation")
        valid_dataset = train_dataset
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # CPU mode
        pin_memory=False
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print("\n🏗️  Building multi-frame model...")
    num_classes = len(train_dataset.label2id)
    model = MultiFrameSignLanguageModel(num_classes=num_classes)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model parameters: {trainable_params:,} / {total_params:,}")
    
    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")
        
        # ========== TRAINING ==========
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            frames = batch['frames'].to(device)  # (B, 5, 3, 224, 224)
            labels = batch['label_id'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward
            optimizer.zero_grad()
            logits = model(frames, input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Metrics
            train_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            # Update progress
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{train_correct/train_total:.4f}"
            })
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # ========== VALIDATION ==========
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validation"):
                frames = batch['frames'].to(device)
                labels = batch['label_id'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                logits = model(frames, input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        val_loss /= len(valid_loader)
        val_acc = val_correct / val_total
        
        # Print epoch results
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_save_path = os.path.join(save_dir, 'best_distilbert_multiframe.pt')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'label2id': train_dataset.label2id,
                'id2label': train_dataset.id2label,
                'config': TRAIN_CONFIG
            }, model_save_path)
            
            print(f"✅ Saved best model: {model_save_path} (Val Acc: {val_acc:.4f})")
    
    # Save training history
    history_path = os.path.join(RESULTS_DIR, 'training_history_multiframe.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {model_save_path}")
    print(f"History saved to: {history_path}")
    
    return model, history


# ==================== INFERENCE ====================
def load_trained_model(model_path):
    """Load trained multi-frame model"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    label2id = checkpoint['label2id']
    num_classes = len(label2id)
    config = checkpoint.get('config', TRAIN_CONFIG)
    
    model = MultiFrameSignLanguageModel(num_classes=num_classes, config=config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['id2label'], config


def predict_video_multiframe(model, video_path, id2label, tokenizer, config=TRAIN_CONFIG):
    """Dự đoán video với trained multi-frame model"""
    device = torch.device(config['device'])
    model = model.to(device)
    
    # Extract multi frames (same as training)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, 0.0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)
            frames.append(frame_tensor)
        
        frame_count += 1
    
    cap.release()
    
    # Padding
    while len(frames) < max_frames:
        frames.append(frames[-1] if frames else torch.zeros(3, config['image_size'], config['image_size']))
    
    # Stack: (1, num_frames, C, H, W)
    frames_tensor = torch.stack(frames[:max_frames]).unsqueeze(0).to(device)
    
    # Dummy text input
    tokens = tokenizer(
        "test",
        max_length=config['max_length'],
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    
    # Predict
    with torch.no_grad():
        logits = model(frames_tensor, input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_id].item()
    
    prediction = id2label[pred_id]
    
    return prediction, confidence


# ==================== MAIN ====================
if __name__ == "__main__":
    print("\nChọn chức năng:")
    print("1. Train model (DistilBERT + Multi-frame)")
    print("2. Test model đã train")
    print("3. Evaluate trên valid dataset")
    print("0. Thoát")
    
    choice = input("\nNhập lựa chọn: ").strip()
    
    if choice == "1":
        epochs = int(input(f"Số epochs (mặc định {TRAIN_CONFIG['epochs']}): ").strip() or TRAIN_CONFIG['epochs'])
        batch_size = int(input(f"Batch size (mặc định {TRAIN_CONFIG['batch_size']}): ").strip() or TRAIN_CONFIG['batch_size'])
        
        print("\n⚙️  Training với Multi-frame:")
        print(f"  - Frames per video: {TRAIN_CONFIG['max_frames']}")
        print(f"  - Sample rate: {TRAIN_CONFIG['frame_sample_rate']}")
        print(f"  - Image size: {TRAIN_CONFIG['image_size']}x{TRAIN_CONFIG['image_size']}")
        
        confirm = input("\nBắt đầu training? (y/n): ").strip().lower()
        if confirm == 'y':
            model, history = train_model(epochs=epochs, batch_size=batch_size)
        
    elif choice == "2":
        model_path = os.path.join(MODELS_DIR, 'best_distilbert_multiframe.pt')
        
        if not os.path.exists(model_path):
            print("❌ Model chưa được train! Chạy option 1 trước.")
        else:
            model, id2label, config = load_trained_model(model_path)
            tokenizer = DistilBertTokenizer.from_pretrained(TRAIN_CONFIG['text_encoder'])
            
            video_path = input("Nhập đường dẫn video: ").strip()
            
            if os.path.exists(video_path):
                print("\n🔄 Analyzing video (multi-frame)...")
                prediction, confidence = predict_video_multiframe(model, video_path, id2label, tokenizer, config)
                print(f"\n✅ Prediction: {prediction}")
                print(f"📊 Confidence: {confidence:.2%}")
            else:
                print("❌ File không tồn tại!")
    
    elif choice == "3":
        print("🚧 Đang phát triển...")
    
    elif choice == "0":
        print("👋 Tạm biệt!")
    
    else:
        print("❌ Lựa chọn không hợp lệ!")