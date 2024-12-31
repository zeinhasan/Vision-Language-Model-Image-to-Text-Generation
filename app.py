# app.py
from flask import Flask, request, jsonify
from PIL import Image
import moondream as md
import os
import requests
import gzip
import shutil
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define constants
MODEL_PATH = "moondream-2b-int8.bin"
COMPRESSED_MODEL_URL = "https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-2b-int8.mf.gz?download=true"
COMPRESSED_MODEL_PATH = "moondream-2b-int8.mf.gz"

def download_and_extract_model(compressed_model_path: str, model_path: str, model_url: str):
    """
    Download and extract the model file if it does not exist.
    """
    if not os.path.exists(model_path):
        if not os.path.exists(compressed_model_path):
            print(f"Model not found. Downloading from {model_url}...")
            response = requests.get(model_url, stream=True)
            if response.status_code == 200:
                with open(compressed_model_path, "wb") as model_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        model_file.write(chunk)
                print(f"Compressed model downloaded successfully to {compressed_model_path}.")
            else:
                raise RuntimeError(f"Failed to download model. HTTP Status: {response.status_code}")
        
        print(f"Extracting model from {compressed_model_path}...")
        with gzip.open(compressed_model_path, 'rb') as compressed_file:
            with open(model_path, 'wb') as model_file:
                shutil.copyfileobj(compressed_file, model_file)
        print(f"Model extracted to {model_path}.")

# Ensure the model is available
download_and_extract_model(COMPRESSED_MODEL_PATH, MODEL_PATH, COMPRESSED_MODEL_URL)

# Load the model
model = md.vl(model=MODEL_PATH)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/caption/', methods=['POST'])
def generate_caption():
    """
    Generate a caption for the uploaded image.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read and process the image
        image = Image.open(file.stream)
        encoded_image = model.encode_image(image)
        caption = model.caption(encoded_image)["caption"]
        
        return jsonify({'caption': caption})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/query/', methods=['POST'])
def ask_question():
    """
    Answer a question based on the uploaded image.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        question = request.form.get('question')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Read and process the image
        image = Image.open(file.stream)
        encoded_image = model.encode_image(image)
        answer = model.query(encoded_image, question)["answer"]
        
        return jsonify({'question': question, 'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)