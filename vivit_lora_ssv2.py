"""
ViViT-B LoRA Fine-Tuning on SSv2
RTX 5090 run: 100% training data, 35 epochs
Metrics: Acc, FT time, Inference latency, Peak Memory, Avg Power, Energy
"""

import os, json, random, re, time, subprocess, threading
import numpy as np
import torch
import torch.nn as nn
import cv2
from torch.utils.data import Dataset, DataLoader
from transformers import VivitForVideoClassification
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
VIDEO_DIR   = '/data/ssv2/videos/20bn-something-something-v2/'   # change if needed
LABELS_DIR  = '/data/ssv2/labels/'                                # change if needed
SAVE_DIR    = './checkpoints/vivit_lora/'
RESULTS_DIR = './results/'
NUM_EPOCHS  = 35
BATCH_SIZE  = 16
LR          = 1e-4
NUM_FRAMES  = 32
NUM_WORKERS = 4
LORA_R      = 16
LORA_ALPHA  = 32

os.makedirs(SAVE_DIR,    exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(f'GPU:  {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')


# ── Power Monitor ──────────────────────────────────────────────────────────
class PowerMonitor:
    def __init__(self):
        self.readings = []
        self._stop    = False
        self._thread  = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self._stop:
            try:
                out = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                    stderr=subprocess.DEVNULL).decode().strip()
                self.readings.append(float(out))
            except:
                pass
            time.sleep(1)

    def start(self): self._thread.start()

    def stop(self):
        self._stop = True
        self._thread.join(timeout=3)

    def avg_power(self):
        return sum(self.readings) / len(self.readings) if self.readings else 0.0


# ── Dataset ────────────────────────────────────────────────────────────────
def template_to_label(template):
    return re.sub(r'\[.*?\]', 'something', template).strip()

def load_video_cv2(path, num_frames=32):
    try:
        cap   = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release(); return None
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames  = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()
        if len(frames) < num_frames // 2: return None
        while len(frames) < num_frames: frames.append(frames[-1])
        return np.stack(frames[:num_frames])
    except:
        return None

class SSv2Dataset(Dataset):
    def __init__(self, data, label2id, video_dir, num_frames=32):
        self.video_dir  = video_dir
        self.num_frames = num_frames
        self.valid      = []
        self.MEAN = np.array([0.485, 0.456, 0.406])
        self.STD  = np.array([0.229, 0.224, 0.225])
        for item in data:
            label_name = template_to_label(item['template'])
            label_id   = label2id.get(label_name, -1)
            if label_id == -1: continue
            for ext in ['.webm', '.mp4', '']:
                path = os.path.join(video_dir, str(item['id']) + ext)
                if os.path.exists(path):
                    self.valid.append((path, label_id)); break
        print(f'Valid videos: {len(self.valid)} / {len(data)}')

    def __len__(self): return len(self.valid)

    def __getitem__(self, idx):
        path, label_id = self.valid[idx]
        frames = load_video_cv2(path, self.num_frames)
        if frames is None: return None
        frames = frames.astype(np.float32) / 255.0
        frames = (frames - self.MEAN) / self.STD
        pixel_values = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # (T,C,H,W)
        return pixel_values, label_id

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None, None
    return torch.stack([b[0] for b in batch]), torch.tensor([b[1] for b in batch])


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    # Load labels
    with open(os.path.join(LABELS_DIR, 'labels.json'))     as f: labels_raw = json.load(f)
    with open(os.path.join(LABELS_DIR, 'train.json'))      as f: train_data = json.load(f)
    with open(os.path.join(LABELS_DIR, 'validation.json')) as f: val_data   = json.load(f)

    label2id = {k: int(v) for k, v in labels_raw.items()}
    print(f'Classes: {len(label2id)} | Train: {len(train_data)} | Val: {len(val_data)}')

    # Datasets
    random.shuffle(train_data)
    train_dataset = SSv2Dataset(train_data, label2id, VIDEO_DIR, NUM_FRAMES)
    val_dataset   = SSv2Dataset(val_data,   label2id, VIDEO_DIR, NUM_FRAMES)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, collate_fn=collate_fn)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, collate_fn=collate_fn)

    # Model — LoRA
    torch.cuda.empty_cache()
    base_model = VivitForVideoClassification.from_pretrained('google/vivit-b-16x2-kinetics400')
    in_features = base_model.classifier.in_features
    base_model.classifier = nn.Linear(in_features, 174)
    nn.init.xavier_uniform_(base_model.classifier.weight)
    nn.init.zeros_(base_model.classifier.bias)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model = model.to(device)
    model.print_trainable_parameters()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # ── Training ──
    epoch_log = []
    pm = PowerMonitor(); pm.start()
    torch.cuda.reset_peak_memory_stats(device)
    ft_start = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        loss_sum, correct, total_n = 0, 0, 0
        for pixels, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}'):
            if pixels is None: continue
            pixels, labels = pixels.to(device), labels.to(device)
            logits = model(pixel_values=pixels).logits
            loss   = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item()
            correct  += (logits.argmax(-1) == labels).sum().item()
            total_n  += labels.size(0)
        acc = 100 * correct / total_n if total_n > 0 else 0
        epoch_log.append({'epoch': epoch+1, 'loss': loss_sum/len(train_loader), 'train_acc': acc})
        print(f'Epoch {epoch+1} | Loss: {loss_sum/len(train_loader):.4f} | Acc: {acc:.2f}%')

        # Save checkpoint every 5 epochs
        if (epoch+1) % 5 == 0:
            ckpt_path = os.path.join(SAVE_DIR, f'vivit_lora_ep{epoch+1}')
            model.save_pretrained(ckpt_path)
            print(f'  Checkpoint saved: {ckpt_path}')

    ft_time_sec = time.time() - ft_start
    pm.stop()
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
    avg_power_w = pm.avg_power()

    print(f'\n=== Training Metrics ===')
    print(f'FT Time:     {ft_time_sec/3600:.2f} h  ({ft_time_sec:.0f} s)')
    print(f'Peak Memory: {peak_mem_gb:.2f} GB')
    print(f'Avg Power:   {avg_power_w:.1f} W')
    print(f'Energy (FT): {avg_power_w * ft_time_sec / 3600:.2f} Wh')

    # ── Inference + Validation ──
    model.eval()
    top1_correct, top5_correct, total_n = 0, 0, 0
    latencies = []
    pm2 = PowerMonitor(); pm2.start()
    inf_start = time.time()

    with torch.no_grad():
        for pixels, labels in tqdm(val_loader, desc='Evaluating'):
            if pixels is None: continue
            pixels, labels = pixels.to(device), labels.to(device)
            t0     = time.time()
            logits = model(pixel_values=pixels).logits
            torch.cuda.synchronize()
            latencies.append((time.time() - t0) / pixels.shape[0])
            top1_correct += (logits.argmax(-1) == labels).sum().item()
            top5_preds    = logits.topk(5, dim=-1).indices
            for i, lbl in enumerate(labels):
                if lbl.item() in top5_preds[i].tolist(): top5_correct += 1
            total_n += labels.size(0)

    inf_time_sec = time.time() - inf_start
    pm2.stop()

    top1        = 100 * top1_correct / total_n
    top5        = 100 * top5_correct / total_n
    avg_inf_lat = sum(latencies) / len(latencies) * 1000
    avg_inf_pow = pm2.avg_power()
    inf_energy  = avg_inf_pow * inf_time_sec / 3600

    # ── Results ──
    print(f'\n{"="*55}')
    print(f'  MODEL           : ViViT-B LoRA (r={LORA_R}, alpha={LORA_ALPHA})')
    print(f'  Epochs          : {NUM_EPOCHS}  |  Train clips: ALL')
    print(f'{"="*55}')
    print(f'  Top-1 Accuracy  : {top1:.2f}%')
    print(f'  Top-5 Accuracy  : {top5:.2f}%')
    print(f'  FT Time         : {ft_time_sec/3600:.2f} h')
    print(f'  Inf Latency     : {avg_inf_lat:.1f} ms/clip')
    print(f'  Peak Memory     : {peak_mem_gb:.2f} GB')
    print(f'  Avg Power (FT)  : {avg_power_w:.1f} W')
    print(f'  Energy (FT)     : {avg_power_w * ft_time_sec / 3600:.2f} Wh')
    print(f'  Avg Power (Inf) : {avg_inf_pow:.1f} W')
    print(f'  Energy (Inf)    : {inf_energy:.4f} Wh')
    print(f'{"="*55}')

    results = {
        'model': 'ViViT-B LoRA', 'dataset': 'SSv2',
        'epochs': NUM_EPOCHS, 'batch_size': BATCH_SIZE,
        'lora_r': LORA_R, 'lora_alpha': LORA_ALPHA,
        'top1': top1, 'top5': top5,
        'ft_time_sec': ft_time_sec, 'ft_time_h': ft_time_sec/3600,
        'inf_latency_ms_per_clip': avg_inf_lat,
        'peak_memory_gb': peak_mem_gb,
        'avg_power_train_w': avg_power_w,
        'energy_train_wh': avg_power_w * ft_time_sec / 3600,
        'avg_power_inf_w': avg_inf_pow,
        'energy_inf_wh': inf_energy,
        'epoch_log': epoch_log
    }
    with open(os.path.join(RESULTS_DIR, 'vivit_lora_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('Results saved!')

if __name__ == '__main__':
    main()
