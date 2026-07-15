#!/usr/bin/env bash
# Fully-offline smoke: runs the demo query against the mock brain (:9000).
# No cluster required. Usage:  ./dev_mock.sh
set -euo pipefail
cd "$(dirname "$0")"
source team.env
export LITELLM_BASE_URL="http://localhost:9000/v1"   # override -> mock brain
export DOMAIN_PREDICT_MODE="mock"
[ -d .venv ] && source .venv/bin/activate
exec python -m demo.run_demo --mock
