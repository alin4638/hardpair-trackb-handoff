#!/usr/bin/env bash
# Download NCT-CRC-HE-100K + CRC-VAL-HE-7K (Zenodo 1214456, CC-BY-4.0) with parallel byte ranges,
# verify MD5 against the Zenodo record, then unzip into data/raw/.
# Zenodo throttles per connection (~2.7 MB/s), so each zip is fetched as N ranged parts.
# Re-running skips finished parts/zips.   Usage: bash scripts/download_data.sh [N_PARTS]
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-16}
Z=data/raw/zips; mkdir -p "$Z"

fetch() {  # name md5 size
  local name=$1 md5=$2 size=$3 url="https://zenodo.org/api/records/1214456/files/$1/content"
  local out="$Z/$name" parts="$Z/$name.parts"
  if [ -f "$out" ] && [ "$(md5sum "$out" | cut -d' ' -f1)" = "$md5" ]; then echo "[ok] $name already verified"; return; fi
  mkdir -p "$parts"; local chunk=$(( (size + N - 1) / N ))
  for i in $(seq 0 $((N - 1))); do
    local a=$((i * chunk)) b=$(((i + 1) * chunk - 1)); [ $b -ge $size ] && b=$((size - 1))
    local p="$parts/$(printf %03d $i)" want=$((b - a + 1))
    [ -f "$p" ] && [ "$(stat -c %s "$p")" = "$want" ] && continue
    ( for t in 1 2 3 4 5; do curl -sfL --retry 5 -r "$a-$b" -o "$p" "$url" && [ "$(stat -c %s "$p")" = "$want" ] && exit 0; sleep 5; done; echo "part $i failed"; exit 1 ) &
  done
  wait
  cat "$parts"/* > "$out"
  local got; got=$(md5sum "$out" | cut -d' ' -f1)
  [ "$got" = "$md5" ] || { echo "[FAIL] $name md5 $got != $md5"; exit 1; }
  rm -rf "$parts"; echo "[ok] $name md5 verified"
}

t0=$(date +%s)
fetch CRC-VAL-HE-7K.zip       2fd1651b4f94ebd818ebf90ad2b6ce06 $(curl -sIL https://zenodo.org/api/records/1214456/files/CRC-VAL-HE-7K.zip/content | grep -i '^content-length' | tail -1 | tr -dc 0-9)
fetch NCT-CRC-HE-100K.zip     6fd702d11df6292bc054397ae038a464 $(curl -sIL https://zenodo.org/api/records/1214456/files/NCT-CRC-HE-100K.zip/content | grep -i '^content-length' | tail -1 | tr -dc 0-9)
echo "downloads done in $(( $(date +%s) - t0 ))s"

for z in CRC-VAL-HE-7K NCT-CRC-HE-100K; do
  [ -d "data/raw/$z" ] && echo "[ok] data/raw/$z already unzipped" && continue
  unzip -q "$Z/$z.zip" -d data/raw/ && echo "[ok] unzipped $z"
done
echo "all done in $(( $(date +%s) - t0 ))s"
