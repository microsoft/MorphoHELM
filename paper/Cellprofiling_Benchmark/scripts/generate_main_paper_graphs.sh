#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

exec python "${ROOT}/paper/Cellprofiling_Benchmark/scripts/generate_paper_graphs.py" "$@"
