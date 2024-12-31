from fastapi import FastAPI, UploadFile, Form, HTTPException
from PIL import Image
import moondream as md
import os
import requests
import gzip
import shutil
import uvicorn
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify a list of origins)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

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
        
        # Extract the model
        print(f"Extracting model from {compressed_model_path}...")
        with gzip.open(compressed_model_path, 'rb') as compressed_file:
            with open(model_path, 'wb') as model_file:
                shutil.copyfileobj(compressed_file, model_file)
        print(f"Model extracted to {model_path}.")

# Ensure the model is available
download_and_extract_model(COMPRESSED_MODEL_PATH, MODEL_PATH, COMPRESSED_MODEL_URL)

# Load the model
model = md.vl(model=MODEL_PATH)

@app.post("/caption/")
async def generate_caption(file: UploadFile):
    """
    Generate a caption for the uploaded image.
    """
    try:
        # Load and encode the image
        image = Image.open(file.file)
        encoded_image = model.encode_image(image)

        # Generate a caption
        caption = model.caption(encoded_image)["caption"]
        return {"caption": caption}
    except Exception as e:
        return {"error": str(e)}


@app.post("/query/")
async def ask_question(file: UploadFile, question: str = Form(...)):
    """
    Answer a question based on the uploaded image.
    """
    try:
        # Load and encode the image
        image = Image.open(file.file)
        encoded_image = model.encode_image(image)

        # Answer the question
        answer = model.query(encoded_image, question)["answer"]
        return JSONResponse(content={"question": question, "answer": answer})
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)