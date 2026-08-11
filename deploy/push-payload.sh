#!/usr/bin/env bash
# Send the weights and the collected data to the server.
#
# `git pull` brings none of this: models/ and data/ are not versioned, on
# purpose, and together they are 1.9 GB. This is what fills the gap, and it is
# run from the workstation, not from the server.
#
#   ./deploy/push-payload.sh user@vps                    # the minimum, ~490 MB
#   ./deploy/push-payload.sh user@vps --all              # every model, ~1.9 GB
#   ./deploy/push-payload.sh user@vps --dest /srv/factnet
#
# rsync only sends what differs, so running it again after a retrain moves just
# the model that changed.
set -euo pipefail

TARGET=${1:-}
DEST=/opt/factnet
ALL=0

shift || true
while [ $# -gt 0 ]; do
  case "$1" in
    --all)  ALL=1 ;;
    --dest) DEST=$2; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ -z "$TARGET" ]; then
  echo "usage: $0 user@host [--all] [--dest /opt/factnet]" >&2
  exit 2
fi

cd "$(dirname "$0")/.."

# The dashboard reports missing models rather than failing, so the minimum set
# is a real option: one transformer, the linear baseline and the graph models.
# The other three transformers only populate the model selector.
if [ "$ALL" -eq 1 ]; then
  MODELS=(models/)
else
  MODELS=(models/roberta-liar models/tfidf.joblib models/graph models/manifest.json)
fi

for path in "${MODELS[@]}" data/samples.jsonl; do
  [ -e "$path" ] || { echo "missing locally: $path" >&2; exit 1; }
done

# One authentication for the whole transfer. Without this each rsync below
# opens its own connection, which means one prompt each when the server asks
# for anything interactive.
CTL=$(mktemp -u /tmp/factnet-ssh-XXXXXX)
SSH="ssh -o ControlMaster=auto -o ControlPath=$CTL -o ControlPersist=300"
cleanup() { $SSH -O exit "$TARGET" 2>/dev/null || true; }
trap cleanup EXIT

echo "==> destination $TARGET:$DEST"
echo "    (authenticate once; the session is reused for every transfer)"
$SSH "$TARGET" "mkdir -p $DEST/models/graph $DEST/data/raw/bluesky"

echo "==> models"
rsync -avh --progress --partial -e "$SSH" --relative "${MODELS[@]}" "$TARGET:$DEST/"

echo "==> the four stored cascades"
rsync -avh --partial -e "$SSH" --relative data/samples.jsonl "$TARGET:$DEST/"

# The anonymised export, not the working file: the server has no need for the
# records that still carry account identities, and this is the one already
# published alongside the article.
if [ -f data/raw/bluesky/cascades-by-source-anonymised.jsonl ]; then
  echo "==> the collected sample, anonymised"
  rsync -avh --progress --partial -e "$SSH" \
    data/raw/bluesky/cascades-by-source-anonymised.jsonl \
    "$TARGET:$DEST/data/raw/bluesky/cascades-by-source.jsonl"
fi

echo
echo "done. On the server:"
echo "    cd $DEST && docker compose up -d --build"
