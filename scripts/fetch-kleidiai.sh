#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="${ROOT}/native/third_party/kleidiai"
PIN="247088200c679f30b1b4a680bd12fee18457a100" # Arm KleidiAI v1.9.0
if [ -d "${DIR}/.git" ] && [ "$(git -C "${DIR}" rev-parse HEAD)" = "${PIN}" ]; then
  echo "Pinned KleidiAI already installed"
  exit 0
fi
rm -rf "${DIR}"
mkdir -p "$(dirname "${DIR}")"
git clone --depth 1 --branch v1.9.0 https://github.com/ARM-software/kleidiai.git "${DIR}"
test "$(git -C "${DIR}" rev-parse HEAD)" = "${PIN}"
