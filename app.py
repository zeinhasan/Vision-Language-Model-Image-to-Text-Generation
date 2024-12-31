# app.py
from flask import Flask, request, jsonify
import moondream as md
from PIL import Image
import io
import os

app = Flask(__name__)

# Initialize model with the local path
MODEL_PATH = "model/moondream-2b-int8.bin"

@app.before_first_request
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = md.vl(model_path=MODEL_PATH)

@app.route('/caption', methods=['POST'])
def generate_caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    
    try:
        caption = model.caption(image)["caption"]
        return jsonify({'caption': caption})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    if 'question' not in request.form:
        return jsonify({'error': 'No question provided'}), 400
    
    image_file = request.files['image']
    question = request.form['question']
    image = Image.open(io.BytesIO(image_file.read()))
    
    try:
        answer = model.query(image, question)["answer"]
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)