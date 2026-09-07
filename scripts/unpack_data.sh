#!/usr/bin/env bash
# Decompress the large network files in place (keeps the .gz originals).
# The analysis scripts read the uncompressed names; the decompressed copies are
# listed in .gitignore so they are never committed.
set -euo pipefail
cd "$(dirname "$0")/../data/networks"
for f in *.gz; do
  out="${f%.gz}"
  if [ -f "$out" ]; then
    echo "exists   $out"
  else
    echo "unpack   $f -> $out"
    gunzip -k "$f"
  fi
done
