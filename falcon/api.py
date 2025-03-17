from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
from ..inference import load_model, generate_text, load_model_cpu

app = Flask(__name__, static_folder='../static')
model = None
tokenizer = None

# Swagger UI integration
SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "Falcon 7B API"}
)
app.register_blueprint(swaggerui_blueprint)

# Use this instead of before_first_request
def initialize_model():
    global model, tokenizer
    if model is None:
        model_id = "distilgpt2"  # CPU-friendly model
        model, tokenizer = load_model_cpu(model_id)

@app.route('/generate', methods=['POST'])
def generate():
    global model, tokenizer
    
    # Initialize model if not already loaded
    initialize_model()
    
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 100)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    
    try:
        generated_text = generate_text(model, tokenizer, prompt, 
                                    max_length, temperature, top_p)
        return jsonify({"generated_text": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)