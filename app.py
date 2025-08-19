# app.py
from flask import Flask, request, jsonify
import torch
import utils
from utils import load_trained_model, preprocess_image, get_class_name_by_id
import config

app = Flask(__name__)

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
            return jsonify({"class_id": class_id, 'class_name': get_class_name_by_id(class_id, 'en')})

        except Exception as e:
            return jsonify({"error": f"An error occurred: {e}"}), 500

@app.route('/classes_info', methods=['GET'])
def get_classes_info():
     #lang = request.args['language'] or 'eng'
     lang = 'en'
     if lang not in config.SUPPORTED_LANGUAGES:
         return jsonify({'Message': f'Language {lang} not supported'}), 400
     classes_info = utils.get_classes_names(lang)
     return jsonify(classes_info)


if __name__ == "__main__":
    # Запуск для разработки. Для production используйте Gunicorn.
    app.run(host="0.0.0.0", port=5000, debug=False)
