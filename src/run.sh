#!/usr/bin/env bash
# Serve the agent /query + /health endpoint on $PORT (default 5000).
# Usage:  ./run.sh
set -euo pipefail
cd "$(dirname "$0")"
source team.env
[ -d .venv ] && source .venv/bin/activate
echo "Serving on :${PORT:-5000}  (brain=$BRAIN_MODEL via $LITELLM_BASE_URL, domain=$DOMAIN_PREDICT_MODE)"
exec python server.py
