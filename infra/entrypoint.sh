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
input)
  log "Preprocessing input..."
  exec python src/input.py "$@"
  ;;
label)
  log "Preprocessing label..."
  exec python src/label.py "$@"
  ;;
debug)
  log "Debugging..."
  exec /bin/bash "$@"
  ;;
*)
  echo "[Error] Unknown command: $COMMAND"
  echo "Try one of: preprocess_input, preprocess_label, debug"
  exit 1
  ;;
esac
