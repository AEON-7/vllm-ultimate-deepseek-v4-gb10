#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 ghcr.io/aeon-7/vllm-ultimate-deepseek-v4-gb10:<tag>" >&2
  exit 2
fi

IMAGE="$1"

echo "Pushing ${IMAGE}"
docker push "${IMAGE}"

cat <<EOF

Pushed: ${IMAGE}

Keep the package private while this remains experimental.
Grant tester access through the private repository/package settings.
EOF

