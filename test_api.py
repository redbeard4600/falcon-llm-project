import requests

try:
    response = requests.post(
        "http://localhost:5000/generate",
        json={"prompt": "What's the future of AI?", "max_length": 100}
    )
    print(response.json())
except Exception as e:
    print(f"Error: {e}")