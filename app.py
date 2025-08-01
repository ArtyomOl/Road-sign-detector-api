# app.py
from flask import Flask, request, jsonify
import torch
from utils import load_trained_model, preprocess_image, get_class_name_by_id

app = Flask(__name__)

# --- Загрузка модели при старте приложения ---
# Модель загружается ОДИН РАЗ, а не при каждом запросе. Это критически важно для производительности.
model = load_trained_model()

@app.route("/")
def index():
    return "Traffic Sign Recognition API is running. Use POST /predict to classify an image."

@app.route("/predict", methods=["POST"])
def predict():
    # 1. Проверка, что файл был отправлен
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    # 2. Проверка, что имя файла не пустое
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        try:
            # 3. Чтение байтов файла
            image_bytes = file.read()
            
            # 4. Предобработка изображения
            input_tensor = preprocess_image(image_bytes)
            
            # 5. Получение предсказания
            with torch.no_grad():
                output = model(input_tensor)
                _, predicted_idx = torch.max(output, 1)
                class_id = predicted_idx.item()
            
            # 6. Формирование и отправка ответа
            return jsonify({"class_id": class_id, 'class_name': get_class_name_by_id(class_id)})

        except Exception as e:
            return jsonify({"error": f"An error occurred: {e}"}), 500

if __name__ == "__main__":
    # Запуск для разработки. Для production используйте Gunicorn.
    app.run(host="0.0.0.0", port=5000, debug=False)
