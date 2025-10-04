#!/bin/bash
set -e

COMMAND="$1"
shift

echo "[Entrypoint] COMMAND = $COMMAND"
echo "[Entrypoint] ARGS    = $@"

log() {
  echo "[$(date +'%Y-%m-%dT%H:%M:%S')] $1"
}

case "$COMMAND" in
train)
  log "Training model..."
  exec python src/train.py "$@"
  ;;
inference)
  log "Running inference..."
  exec python src/inference.py "$@"
  ;;
debug)
  log "Debugging..."
  exec /bin/bash "$@"
  ;;
*)
  echo "[Error] Unknown command: $COMMAND"
  echo "Try one of: train, inference, debug"
  exit 1
  ;;
esac
