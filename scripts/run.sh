# scripts/run.sh
#!/bin/bash
source .venv/bin/activate
python -m falcon.main --prompt "$1"