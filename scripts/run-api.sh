# scripts/run-api.sh
#!/bin/bash
source .venv/bin/activate
export FLASK_APP=falcon.api
flask run --host=0.0.0.0 --port=5000