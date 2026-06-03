"""
Rice Disease Detection API
--------------------------
This application serves a PyTorch deep learning model via a Flask REST API.
It processes uploaded images of rice leaves and predicts the disease.
"""

import os
import io
import logging
import torch
from flask import Flask, request, jsonify, render_template
from PIL import Image
from torchvision import transforms

# Import model architecture
from models import build_model
from Train_Lua import CLASS_NAMES

# --- Configuration & Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Giới hạn dung lượng file upload (16MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

# --- Global Definitions ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoint/lua/best.pt"

# Chuẩn hoá ảnh theo ImageNet chuẩn PyTorch
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

model = None  # Global model instance

def load_ai_model():
    """Khởi tạo và nạp trọng số vào mô hình AI"""
    global model
    logger.info(f"Initializing system on device: {DEVICE.type.upper()}")
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Checkpoint not found at '{MODEL_PATH}'. Please ensure the model is trained.")
        return False

    try:
        logger.info(f"Loading weights from {MODEL_PATH}...")
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        num_classes = checkpoint.get("num_classes", len(CLASS_NAMES))
        
        model_instance = build_model(num_classes, DEVICE)
        
        # Load weights (strict=False hỗ trợ tương thích ngược)
        load_res = model_instance.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if hasattr(load_res, 'missing_keys') and load_res.missing_keys:
            logger.warning(f"Missing keys in state_dict: {len(load_res.missing_keys)}")
            
        model_instance.to(DEVICE)
        model_instance.eval()
        
        model = model_instance
        logger.info("✅ AI Model loaded and ready for inference!")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False

# Nạp model ngay khi khởi động server
load_ai_model()

# --- API Routes ---

@app.route('/')
def index():
    """Render giao diện web chính"""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """API kiểm tra trạng thái sức khỏe của server"""
    status = "healthy" if model is not None else "degraded (model not loaded)"
    return jsonify({
        "status": status,
        "device": DEVICE.type,
        "classes_supported": len(CLASS_NAMES)
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """API tiếp nhận ảnh và trả về kết quả dự đoán"""
    if model is None:
        return jsonify({'error': 'Internal Server Error: AI model is not initialized.'}), 503
        
    if 'file' not in request.files:
        return jsonify({'error': 'Bad Request: No file part in the request.'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Bad Request: No selected file.'}), 400
        
    try:
        # Đọc và tiền xử lý ảnh
        image_bytes = file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = TRANSFORM(image_pil).unsqueeze(0).to(DEVICE)
        
        # Inference - Suy luận
        with torch.no_grad():
            output = model(image_tensor)
            probs = torch.softmax(output, dim=1)
            predicted_prob, predicted_class = torch.max(probs, dim=1)
            
        score = predicted_prob.item() * 100
        class_idx = predicted_class.item()
        
        # Format kết quả
        predicted_label = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"Unknown Class ({class_idx})"
        
        logger.info(f"Predicted: {predicted_label} ({score:.2f}%)")
        
        return jsonify({
            'success': True,
            'label': predicted_label,
            'confidence': f"{score:.2f}%",
            'raw_score': score
        })
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal Server Error: Failed to process the image.'}), 500

# --- Server Execution ---
if __name__ == '__main__':
    logger.info("Starting Flask server...")
    # use_reloader=False giúp model không bị nạp 2 lần khi chạy debug
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
