# flask_api.py
from flask import Flask, request, jsonify
import sys
import os

# Add the project root to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import generate_text
from monitoring import monitor
from dataclasses import asdict

app = Flask(__name__)

@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.json
    
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Prompt is required'}), 400
    
    prompt = data['prompt']
    max_length = data.get('max_length', 100)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 0.9)
    
    try:
        completion, stats = generate_text(
            prompt, 
            max_length=max_length,
            temperature=temperature,
            top_p=top_p
        )
        
        # Convert dataclass to dict for JSON serialization
        stats_dict = asdict(stats)
        stats_dict['timestamp'] = stats_dict['timestamp'].isoformat()
        
        return jsonify({
            'completion': completion,
            'stats': stats_dict
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def api_stats():
    summary = monitor.get_summary_stats()
    return jsonify(summary)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)