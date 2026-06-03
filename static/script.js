document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('image-preview-container');
    const previewImage = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const loading = document.getElementById('loading');
    const resultContainer = document.getElementById('result-container');
    const diseaseName = document.getElementById('disease-name');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceVal = document.getElementById('confidence-val');

    // Mở file dialog khi click vào vùng drop-zone
    dropZone.addEventListener('click', (e) => {
        if (e.target !== removeBtn) {
            fileInput.click();
        }
    });

    // Ngăn chặn hành vi mặc định khi kéo thả
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Hiệu ứng highlight khi kéo file qua
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('dragover');
    }

    function unhighlight(e) {
        dropZone.classList.remove('dragover');
    }

    // Xử lý sự kiện thả file
    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    // Xử lý sự kiện chọn file từ input
    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        const file = files[0];
        
        // Kiểm tra loại file
        if (!file.type.startsWith('image/')) {
            showError("Vui lòng chỉ tải lên file hình ảnh.");
            return;
        }

        previewFile(file);
        uploadFile(file);
    }

    // Hiển thị bản xem trước
    function previewFile(file) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            previewImage.src = reader.result;
            previewContainer.classList.remove('hidden');
            resultContainer.classList.add('hidden');
            
            // Xóa thông báo lỗi cũ nếu có
            const oldError = document.querySelector('.error-msg');
            if (oldError) oldError.remove();
        }
    }

    // Xóa ảnh
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Ngăn sự kiện click lan ra drop-zone
        fileInput.value = '';
        previewImage.src = '';
        previewContainer.classList.add('hidden');
        resultContainer.classList.add('hidden');
        confidenceBar.style.width = '0%';
    });

    // Upload và phân tích ảnh
    function uploadFile(file) {
        const url = '/predict';
        const formData = new FormData();
        formData.append('file', file);

        // Hiển thị loading
        loading.classList.remove('hidden');
        resultContainer.classList.add('hidden');

        fetch(url, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || 'Lỗi kết nối') });
            }
            return response.json();
        })
        .then(data => {
            loading.classList.add('hidden');
            
            if (data.error) {
                showError(data.error);
                return;
            }

            // Hiển thị kết quả
            diseaseName.textContent = data.label;
            
            // Xử lý độ tin cậy để tạo animation cho thanh process bar
            const confidenceFloat = parseFloat(data.confidence);
            confidenceVal.textContent = data.confidence;
            
            resultContainer.classList.remove('hidden');
            
            // Trigger reflow for animation
            setTimeout(() => {
                confidenceBar.style.width = `${confidenceFloat}%`;
                
                // Thay đổi màu tùy theo độ tin cậy
                if (confidenceFloat > 80) {
                    confidenceBar.style.background = "linear-gradient(135deg, #34d399 0%, #059669 100%)";
                } else if (confidenceFloat > 50) {
                    confidenceBar.style.background = "linear-gradient(135deg, #fbbf24 0%, #d97706 100%)";
                } else {
                    confidenceBar.style.background = "linear-gradient(135deg, #f87171 0%, #dc2626 100%)";
                }
            }, 50);

        })
        .catch(error => {
            loading.classList.add('hidden');
            showError("Có lỗi xảy ra: " + error.message);
        });
    }

    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-msg';
        errorDiv.textContent = message;
        
        const oldError = document.querySelector('.error-msg');
        if (oldError) oldError.remove();
        
        dropZone.appendChild(errorDiv);
    }
});
