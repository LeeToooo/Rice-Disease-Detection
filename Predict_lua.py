import torch
import torch.nn as nn
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import argparse
import matplotlib.pyplot as plt
from torchvision.models import resnet50
from dataset_Lua import Lua_dataset, resolve_dataset_root
from models import build_model

import warnings
warnings.filterwarnings("ignore")




# Kiến trúc mô hình đã được chuyển qua import chung từ Train_Lua.py để đảm bảo đồng bộ (ResNet50 + CBAM).


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


# ===================== Args =====================
def get_args():
    parser = argparse.ArgumentParser(description="Dự đoán bệnh lá lúa với VGG16-BN + CBAM")
    parser.add_argument("--image_size", "-i", type=int, default=224,
                        help="Kích thước ảnh đầu vào (default: 224)")
    parser.add_argument("--image_path", "-p", type=str,
                        default="leaf_blast_1.jpg",
                        help="Đường dẫn đến ảnh cần dự đoán")
    parser.add_argument("--checkpoint_path", "-c", type=str,
                        default="checkpoint/lua",
                        help="Đường dẫn đến file checkpoint hoặc thư mục chứa best.pt")
    parser.add_argument("--data_path", "-d", type=str, default=None,
                        help="Thư mục chứa train/ và test/. Nếu không truyền, sẽ tự tìm từ image_path")
    args = parser.parse_args()
    return args


# ===================== Inference =====================
def inference(args):
    # Load checkpoint first so we can infer class count if needed
    checkpoint_file = args.checkpoint_path
    if os.path.isdir(checkpoint_file):
        candidate_paths = [
            os.path.join(checkpoint_file, "best.pt"),
            os.path.join(checkpoint_file, "last.pt"),
            os.path.join(checkpoint_file, "lua", "best.pt"),
            os.path.join(checkpoint_file, "lua", "last.pt"),
        ]
        checkpoint_file = next((p for p in candidate_paths if os.path.exists(p)), None)

    if checkpoint_file is None or not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {args.checkpoint_path}")

    checkpoint = torch.load(checkpoint_file, map_location="cpu")

    # Try to load class names from the training dataset to ensure consistency
    classes = None
    data_root = infer_dataset_root(args.image_path, args.data_path)
    if data_root is not None:
        try:
            train_dataset = Lua_dataset(root=data_root, is_train=True, transforms=None)
            classes = train_dataset.categories
            display_names = [c.replace('_', ' ').replace('-', ' ').title() for c in classes]
            print(f"Loaded {len(classes)} classes from dataset root: {data_root}")
            print(f"Categories: {classes}")
            print(f"Display names: {display_names}")
        except Exception as e:
            print(f"Warning: couldn't load dataset categories from {data_root}: {e}")
    else:
        print("Warning: could not infer dataset root from image_path; falling back to checkpoint metadata")

    if classes is not None:
        num_classes = len(classes)
    else:
        num_classes = checkpoint.get("num_classes", None)
        if num_classes is None:
            model_state = checkpoint.get("model_state_dict", {})
            weight_key = next((k for k in model_state.keys() if k.endswith("_fc.4.weight")), None)
            if weight_key is not None:
                num_classes = model_state[weight_key].shape[0]
            else:
                num_classes = 10
        classes = [f"class_{i}" for i in range(num_classes)]
        display_names = [c.replace('_', ' ').replace('-', ' ').title() for c in classes]
        print(f"Using fallback class names: {classes}")
        print(f"Display names: {display_names}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = build_model(num_classes, device)

    # --- Tiền xử lý ảnh (giống test_transform trong Train_Lua.py) ---
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Dùng PIL để đọc ảnh (nhất quán với dataset_Lua.py)
    image_pil = Image.open(args.image_path).convert("RGB")
    image_tensor = transform(image_pil).unsqueeze(0).to(device)  # shape: [1, 3, H, W]

    # --- Tải mô hình RESNet50 + CBAM ---

    # Checkpoint đã load sẵn ở trên, chỉ cần chuyển về thiết bị
    # và load state dict đã khớp với num_classes
    load_res = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if hasattr(load_res, 'missing_keys') and load_res.missing_keys:
        print("Warning: missing keys when loading state_dict:", load_res.missing_keys)
    if hasattr(load_res, 'unexpected_keys') and load_res.unexpected_keys:
        print("Warning: unexpected keys when loading state_dict:", load_res.unexpected_keys)
    model.to(device)
    model.eval()

    # --- Dự đoán ---
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        predicted_prob, predicted_class = torch.max(probs, dim=1)

    score = predicted_prob.item() * 100
    # Show human-friendly disease name
    try:
        predicted_label = display_names[predicted_class.item()]
    except Exception:
        predicted_label = classes[predicted_class.item()]

    print(f"\nKết quả dự đoán:")
    print(f"  Loại bệnh : {predicted_label}")
    print(f"  Độ tin cậy: {score:.2f}%")

    # --- Hiển thị kết quả ---
    plt.figure(figsize=(6, 6))
    plt.imshow(image_pil)
    plt.title(f"{predicted_label}\nĐộ tin cậy: {score:.2f}%", fontsize=13)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = get_args()
    inference(args)