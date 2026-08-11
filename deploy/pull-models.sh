#!/usr/bin/env bash
# Fetch the weights and the sample data onto the server, over HTTPS, no SSH.
#
# Run this ON THE SERVER, from /opt/factnet, after the image has been built:
#
#     ./deploy/pull-models.sh NayJi7/factnet-models
#     HF_TOKEN=hf_xxx ./deploy/pull-models.sh NayJi7/factnet-models   # private repo
#
# It borrows the container, which already carries huggingface_hub, so nothing
# has to be installed on the host. HF_HUB_OFFLINE is set inside the image so
# that serving never touches the network; it is lifted here, and only here.
#
# The repository holds both halves of what git does not carry:
#   the model weights at the root, and data/ alongside them.
set -euo pipefail

REPO=${1:-}
ROOT=${2:-$(pwd)}
IMAGE=${FACTNET_IMAGE:-factnet-dashboard:latest}

if [ -z "$REPO" ]; then
  echo "usage: $0 <user>/<repo> [destination root]" >&2
  exit 2
fi

mkdir -p "$ROOT/models" "$ROOT/data/raw/bluesky"
echo "==> $REPO -> $ROOT"

# --user root because the destination belongs to the host user, not to the
# unprivileged account the service runs as; the container is discarded after
docker run --rm --user root \
  -e HF_HUB_OFFLINE=0 \
  -e HF_HOME=/tmp/hf \
  -e REPO="$REPO" \
  ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
  -v "$ROOT/models:/out/models" \
  -v "$ROOT/data:/out/data" \
  "$IMAGE" \
  python -c "
import os, shutil, pathlib
from huggingface_hub import snapshot_download

repo = os.environ['REPO']
staged = snapshot_download(repo, repo_type='model', local_dir='/tmp/staged')
staged = pathlib.Path(staged)

# the weights sit at the root of the repository, the sample data under data/
for item in staged.iterdir():
    if item.name in ('.cache', '.gitattributes', 'README.md'):
        continue
    target = pathlib.Path('/out') / ('data' if item.name == 'data' else 'models') / item.name
    if item.name == 'data':
        target = pathlib.Path('/out/data')
    if item.is_dir():
        shutil.copytree(item, target, dirs_exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)
print('placed')
" 2>&1 | tail -20

echo
echo "==> models"
du -sh "$ROOT"/models/* 2>/dev/null | sed 's/^/    /'
echo "==> data"
find "$ROOT/data" -type f -name '*.jsonl' -exec du -sh {} \; 2>/dev/null | sed 's/^/    /'
echo
echo "the service picks these up with no restart:"
echo "    curl -s localhost:\${FACTNET_PORT:-8347}/api/health"
