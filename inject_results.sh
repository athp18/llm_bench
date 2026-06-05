#!/usr/bin/env bash
# Embed results into dashboard.html so it can be opened directly in a browser.
# Usage: ./inject_results.sh && open dashboard.html

set -euo pipefail

RESULTS_DIR="$(dirname "$0")/results"
DASHBOARD="$(dirname "$0")/dashboard.html"

data=$(python3 -c "
import json, sys
from pathlib import Path
results = []
for p in sorted(Path('$RESULTS_DIR').glob('*.json')):
    if p.name == 'sweep_summary.json': continue
    results.append(json.loads(p.read_text()))
print(json.dumps(results))
sys.stdout.flush()
")

count=$(echo "$data" | python3 -c "import json,sys; print(len(json.load(sys.stdin)))")

python3 -c "
import re, sys
html = open('$DASHBOARD').read()
html = re.sub(r'const RUNS = \[.*?\];', 'const RUNS = $data;', html, flags=re.DOTALL)
open('$DASHBOARD', 'w').write(html)
"

echo "Injected $count runs into dashboard.html"
open dashboard.html
