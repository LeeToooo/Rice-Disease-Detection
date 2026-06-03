import torch
import torch.nn as nn
import os
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import argparse
import shutil
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_score, recall_score,
                             f1_score, average_precision_score)
from sklearn.preprocessing import label_binarize
import time
import warnings
from datetime import datetime

from dataset_Lua import Lua_dataset, resolve_dataset_root
from torchvision.models import resnet50, ResNet50_Weights
from models import build_model, CBAM, ChannelAttention, SpatialAttention

warnings.filterwarnings("ignore")


# ===================== Seed =====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===================== Models =====================
# Lớp mô hình đã được chuyển sang file models.py để chia việc cho 4 thành viên.


# ===================== Args =====================
def get_args():
    parser = argparse.ArgumentParser(description="Train ResNet50 + CBAM for rice leaf disease")
    parser.add_argument("--data_path",       "-d", type=str,   default=".")
    parser.add_argument("--epochs",          "-e", type=int,   default=100)
    parser.add_argument("--batch_size",      "-b", type=int,   default=32)
    parser.add_argument("--image_size",      "-i", type=int,   default=224)
    parser.add_argument("--lr",              "-l", type=float, default=0.001)
    parser.add_argument("--log_path",        "-p", type=str,   default="TensorBoard/lua")
    parser.add_argument("--checkpoint_path", "-c", type=str,   default="checkpoint/lua")
    parser.add_argument("--predict_image",        type=str,   default=None,  help="Đường dẫn ảnh bên ngoài để dự đoán khi chạy.")
    parser.add_argument("--predict_only",         action="store_true", help="Chỉ chạy inference mà không train nếu có --predict_image.")
    parser.add_argument("--patience",              type=int,   default=15,    help="Early stopping patience")
    parser.add_argument("--num_classes",           type=int,   default=None,  help="Số class (tự detect nếu để None)")
    parser.add_argument("--seed",                  type=int,   default=42)
    parser.add_argument("--warmup_epochs",         type=int,   default=5,     help="Số epoch warmup LR")
    parser.add_argument("--num_workers",           type=int,   default=4)
    args = parser.parse_args()
    return args


# ===================== Confusion Matrix =====================
def plot_confusion_matrix(writer, cm, class_names, epoch):
    figure = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap="cool")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1  # Tránh chia cho 0
    cm_norm = np.around(cm.astype('float') / row_sums, decimals=2)
    threshold = cm_norm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] > threshold else "black"
            plt.text(j, i, cm_norm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)
    plt.close(figure)


# Hàm build_model đã được chuyển sang models.py


def infer_dataset_root(image_path, explicit_root=None):
    if explicit_root:
        found = resolve_dataset_root(explicit_root)
        if found is not None:
            return found

    path = os.path.abspath(image_path)
    current = os.path.dirname(path)
    while current and current != os.path.dirname(current):
        found = resolve_dataset_root(current)
        if found is not None:
            return found
        current = os.path.dirname(current)
    return None


def find_checkpoint_file(checkpoint_path):
    if os.path.isdir(checkpoint_path):
        candidate_paths = [
            os.path.join(checkpoint_path, "best.pt"),
            os.path.join(checkpoint_path, "last.pt"),
            os.path.join(checkpoint_path, "lua", "best.pt"),
            os.path.join(checkpoint_path, "lua", "last.pt"),
        ]
        return next((p for p in candidate_paths if os.path.exists(p)), None)
    return checkpoint_path if os.path.exists(checkpoint_path) else None


def predict_image(args):
    if args.predict_image is None:
        raise ValueError("predict_image phải được cung cấp khi gọi predict_image().")

    checkpoint_file = find_checkpoint_file(args.checkpoint_path)
    if checkpoint_file is None:
        raise FileNotFoundError(f"Không tìm thấy checkpoint tại: {args.checkpoint_path}")

    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    data_root = infer_dataset_root(args.predict_image, args.data_path)
    classes = None

    if data_root is not None:
        try:
            train_dataset = Lua_dataset(root=data_root, is_train=True, transforms=None)
            classes = train_dataset.categories
            print(f"Loaded {len(classes)} classes from dataset root: {data_root}")
            print(f"Categories: {classes}")
        except Exception as e:
            print(f"Warning: không thể load dataset categories từ {data_root}: {e}")

    if classes is not None:
        num_classes = len(classes)
    else:
        num_classes = checkpoint.get("num_classes", None)
        if num_classes is None:
            model_state = checkpoint.get("model_state_dict", {})
            # Tìm class size từ layer cuối cùng trong block sequential
            weight_key = next((k for k in model_state.keys() if k.endswith("fc.6.weight")), None)
            if weight_key is not None:
                num_classes = model_state[weight_key].shape[0]
            else:
                num_classes = 10
        classes = [f"class_{i}" for i in range(num_classes)]

    display_names = [c.replace('_', ' ').replace('-', ' ').title() for c in classes]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(num_classes, device)
    load_res = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if hasattr(load_res, 'missing_keys') and load_res.missing_keys:
        print("Warning: missing keys when loading state_dict:", load_res.missing_keys)
    if hasattr(load_res, 'unexpected_keys') and load_res.unexpected_keys:
        print("Warning: unexpected keys when loading state_dict:", load_res.unexpected_keys)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_pil = Image.open(args.predict_image).convert("RGB")
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        predicted_prob, predicted_class = torch.max(probs, dim=1)

    score = predicted_prob.item() * 100
    predicted_label = display_names[predicted_class.item()]

    print(f"\nKết quả dự đoán:")
    print(f"  Loại bệnh : {predicted_label}")
    print(f"  Độ tin cậy: {score:.2f}%")

    plt.figure(figsize=(6, 6))
    plt.imshow(image_pil)
    plt.title(f"{predicted_label}\nĐộ tin cậy: {score:.2f}%", fontsize=13)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# ===================== Train =====================
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Transforms ----
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(args.image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ---- Datasets & Loaders ----
    train_dataset = Lua_dataset(root=args.data_path, is_train=True,  transforms=train_transform)
    test_dataset  = Lua_dataset(root=args.data_path, is_train=False, transforms=test_transform,
                                categories=train_dataset.categories)

    # Tự detect số class nếu không truyền vào
    num_classes = args.num_classes if args.num_classes else len(train_dataset.categories)
    print(f"Số class: {num_classes} — {train_dataset.categories}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, pin_memory=pin_memory)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, drop_last=False, pin_memory=pin_memory)

    # ---- Model ----
    model = build_model(num_classes, device)

    # ---- Loss / Optimizer / Scheduler ----
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Warmup (LinearLR) + Cosine Annealing
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs]
    )

    # Mixed Precision scaler
    use_amp = device.type == "cuda"
    scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

    # ---- TensorBoard ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_path, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard log: {log_dir}")

    # ---- Checkpoint dir ----
    os.makedirs(args.checkpoint_path, exist_ok=True)

    best_accuracy    = 0.0
    patience_counter = 0
    categories       = train_dataset.categories

    # ===================== Training Loop =====================
    for epoch in range(args.epochs):
        # ---- TRAIN ----
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", colour="cyan")

        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}",
                                     lr=f"{optimizer.param_groups[0]['lr']:.6f}")

        train_loss = running_loss / len(train_loader)
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/LR",   optimizer.param_groups[0]['lr'], epoch)

        # ---- VALIDATION ----
        model.eval()
        all_losses = []
        all_labels = []
        all_preds  = []
        all_probs  = []

        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
                all_losses.append(loss.item())

                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()

        all_probs      = np.concatenate(all_probs)
        all_labels_bin = label_binarize(all_labels, classes=range(num_classes))

        # ---- Metrics ----
        val_loss  = np.mean(all_losses)
        accuracy  = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall    = recall_score(all_labels, all_preds,    average='macro', zero_division=0)
        f1        = f1_score(all_labels, all_preds,        average='macro', zero_division=0)
        map_score = average_precision_score(all_labels_bin, all_probs, average='macro')

        total_images   = len(test_dataset)
        inference_time = (end_time - start_time) / total_images if total_images > 0 else 0
        fps            = 1.0 / inference_time if inference_time > 0 else 0

        # ---- Logging ----
        print(f"\nEpoch [{epoch+1}/{args.epochs}]  LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Train Loss : {train_loss:.4f}  |  Val Loss : {val_loss:.4f}")
        print(f"  Accuracy   : {accuracy:.4f}  |  Precision: {precision:.4f}")
        print(f"  Recall     : {recall:.4f}  |  F1       : {f1:.4f}  |  mAP: {map_score:.4f}")
        print(f"  Inference  : {inference_time*1000:.3f} ms/img  |  FPS: {fps:.1f}")

        writer.add_scalar("Val/Loss",           val_loss,       epoch)
        writer.add_scalar("Val/Accuracy",       accuracy,       epoch)
        writer.add_scalar("Val/Precision",      precision,      epoch)
        writer.add_scalar("Val/Recall",         recall,         epoch)
        writer.add_scalar("Val/F1",             f1,             epoch)
        writer.add_scalar("Val/mAP",            map_score,      epoch)
        writer.add_scalar("Val/Inference_ms",   inference_time * 1000, epoch)
        writer.add_scalar("Val/FPS",            fps,            epoch)

        conf_matrix = confusion_matrix(all_labels, all_preds)
        plot_confusion_matrix(writer, conf_matrix, categories, epoch)

        # ---- Scheduler step (trước early stopping) ----
        scheduler.step()

        # ---- Save checkpoint ----
        checkpoint = {
            "epoch":               epoch,
            "model_state_dict":    model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_accuracy":       best_accuracy,
            "patience_counter":    patience_counter,
            "num_classes":         num_classes,
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_path, "last.pt"))

        if accuracy > best_accuracy:
            best_accuracy    = accuracy
            patience_counter = 0
            torch.save(checkpoint, os.path.join(args.checkpoint_path, "best.pt"))
            print(f"  → Best model saved! Accuracy: {best_accuracy:.4f}")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{args.patience}")

        # ---- Early stopping ----
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered sau {epoch+1} epochs.")
            break

    writer.close()
    print(f"\nTraining hoàn tất! Best accuracy: {best_accuracy:.4f}")
    print(f"Best model: {os.path.join(args.checkpoint_path, 'best.pt')}")


# ===================== Entry Point =====================
if __name__ == "__main__":
    args = get_args()
    if args.predict_image:
        predict_image(args)
    elif args.predict_only:
        raise ValueError("--predict_only yêu cầu phải có --predict_image để chạy inference.")
    else:
        train(args)