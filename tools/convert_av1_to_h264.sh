#!/bin/bash
# Convert AV1 videos to H.264 for faster decord GPU decoding.
# Usage: bash tools/convert_av1_to_h264.sh /path/to/lerobot_dataset /path/to/output
#
# Creates a mirror of the dataset with H.264 videos, copies everything else as-is.
# Uses ffmpeg with libx264 CRF 18 (visually lossless).

set -euo pipefail

SRC="${1:?Usage: $0 <source_dir> <output_dir>}"
DST="${2:?Usage: $0 <source_dir> <output_dir>}"
JOBS="${3:-8}"  # parallel ffmpeg jobs

echo "=== AV1 → H.264 Conversion ==="
echo "Source: ${SRC}"
echo "Output: ${DST}"
echo "Parallel jobs: ${JOBS}"

for ds_dir in "${SRC}"/*/; do
    ds_name=$(basename "${ds_dir}")
    echo
    echo "--- ${ds_name} ---"

    # Copy non-video files
    mkdir -p "${DST}/${ds_name}"
    for sub in data meta; do
        if [ -d "${ds_dir}${sub}" ]; then
            cp -ru "${ds_dir}${sub}" "${DST}/${ds_name}/"
        fi
    done

    # Convert videos
    video_dir="${ds_dir}videos"
    if [ ! -d "${video_dir}" ]; then
        echo "  No videos/ directory, skipping"
        continue
    fi

    find "${video_dir}" -name "*.mp4" | while read -r src_mp4; do
        rel_path="${src_mp4#${ds_dir}}"
        dst_mp4="${DST}/${ds_name}/${rel_path}"
        mkdir -p "$(dirname "${dst_mp4}")"

        if [ -f "${dst_mp4}" ]; then
            continue  # skip existing
        fi

        echo "${dst_mp4}"
    done | xargs -P "${JOBS}" -I {} bash -c '
        src_mp4=$(echo "{}" | sed "s|'"${DST}"'|'"${SRC}"'|; s|/'"${ds_name}"'/|/'"${ds_name}"'/|")
        # Reconstruct source path
        rel=$(echo "{}" | sed "s|^'"${DST}/${ds_name}/"'||")
        src_mp4="'"${ds_dir}"'${rel}"
        ffmpeg -y -i "${src_mp4}" -c:v libx264 -crf 18 -preset fast -an "{}" -loglevel error 2>&1
    '

    # Count converted
    n_converted=$(find "${DST}/${ds_name}/videos" -name "*.mp4" 2>/dev/null | wc -l)
    n_source=$(find "${video_dir}" -name "*.mp4" | wc -l)
    echo "  ${n_converted}/${n_source} videos converted"

    # Update info.json codec
    if [ -f "${DST}/${ds_name}/meta/info.json" ]; then
        python3 -c "
import json
info = json.load(open('${DST}/${ds_name}/meta/info.json'))
for k, v in info['features'].items():
    if v.get('dtype') == 'video' and 'info' in v:
        v['info']['video.codec'] = 'h264'
json.dump(info, open('${DST}/${ds_name}/meta/info.json', 'w'), indent=4)
" 2>/dev/null
        echo "  info.json updated (codec → h264)"
    fi
done

echo
echo "=== Conversion complete ==="
