from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import torch
import requests
from io import BytesIO

app = Flask(__name__)

# Setup device and models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    "gokaygokay/Florence-2-SD3-Captioner", 
    trust_remote_code=True,
    use_auth_token="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"
).to(device).eval()

processor = AutoProcessor.from_pretrained(
    "gokaygokay/Florence-2-SD3-Captioner", 
    trust_remote_code=True,
    use_auth_token="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"
)

def run_example(task_prompt, text_input, image):
    prompt = task_prompt + text_input
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed_answer

@app.route('/describe', methods=['POST'])
def describe_image():
    if 'image_url' not in request.json:
        return jsonify({"error": "Missing 'image_url' in request"}), 400

    image_url = request.json['image_url']
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        task_prompt = "<DESCRIPTION>"
        text_input = "Describe this image in great detail."
        answer = run_example(task_prompt, text_input, image)
        return jsonify({"description": answer.get('<DESCRIPTION>', 'No description available')})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
