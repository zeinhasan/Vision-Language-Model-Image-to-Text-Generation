from fastapi import FastAPI, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import moondream as md
import os
import requests
import gzip
import shutil
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define constants
MODEL_PATH = "moondream-2b-int8.bin"
COMPRESSED_MODEL_URL = "https://huggingface.co/vikhyatk/moondream2/resolve/9dddae84d54db4ac56fe37817aeaeb502ed083e2/moondream-2b-int8.mf.gz?download=true"
COMPRESSED_MODEL_PATH = "moondream-2b-int8.mf.gz"

def download_and_extract_model(compressed_model_path: str, model_path: str, model_url: str):
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

@app.post("/caption/")
async def generate_caption(request: Request, file: UploadFile):
    """
    Generate a caption for the uploaded image with redirect handling.
    """
    try:
        # Store the file in memory
        contents = await file.read()
        # Reset file pointer for potential reuse
        await file.seek(0)
        
        # Check if this is a redirect
        if request.headers.get("X-Forwarded-Host") or request.headers.get("X-Forwarded-Proto"):
            # Handle the redirect by processing the file directly
            image = Image.open(contents)
        else:
            # Normal flow
            image = Image.open(file.file)
            
        encoded_image = model.encode_image(image)
        caption = model.caption(encoded_image)["caption"]
        return {"caption": caption}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/")
async def ask_question(request: Request, file: UploadFile, question: str = Form(...)):
    """
    Answer a question based on the uploaded image with redirect handling.
    """
    try:
        # Store the file in memory
        contents = await file.read()
        # Reset file pointer for potential reuse
        await file.seek(0)
        
        # Check if this is a redirect
        if request.headers.get("X-Forwarded-Host") or request.headers.get("X-Forwarded-Proto"):
            # Handle the redirect by processing the file directly
            image = Image.open(contents)
        else:
            # Normal flow
            image = Image.open(file.file)
            
        encoded_image = model.encode_image(image)
        answer = model.query(encoded_image, question)["answer"]
        return JSONResponse(content={"question": question, "answer": answer})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error answering question: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)