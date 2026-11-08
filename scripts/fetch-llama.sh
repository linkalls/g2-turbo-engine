#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="${ROOT}/native/third_party/llama.cpp"
PIN="e14a779719652200cec7ad58dfa4820edc347182" # llama.cpp 2026-11-06, known PLaMo 3 compatible
if [ -d "${DIR}/.git" ] && [ "$(git -C "${DIR}" rev-parse HEAD)" = "${PIN}" ]; then
  echo "Pinned llama.cpp already fetched"
  exit 0
fi
rm -rf "${DIR}"
mkdir -p "$(dirname "${DIR}")"
git init -q "${DIR}"
git -C "${DIR}" remote add origin https://github.com/ggml-org/llama.cpp.git
git -C "${DIR}" fetch --quiet --filter=blob:none --depth 1 origin "${PIN}"
git -C "${DIR}" -c advice.detachedHead=false checkout --quiet FETCH_HEAD
test "$(git -C "${DIR}" rev-parse HEAD)" = "${PIN}"
