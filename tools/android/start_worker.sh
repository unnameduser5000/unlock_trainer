#!/usr/bin/env bash
set -euo pipefail

serial=""
coordinator_host="${SID_COORDINATOR_HOST:-}"
coordinator_port="${SID_COORDINATOR_PORT:-50051}"
device_id="${SID_DEVICE_ID:-}"
local_port="${SID_LOCAL_PORT:-26052}"
install=false

usage() {
  cat <<'EOF'
Usage: tools/android/start_worker.sh --coordinator-host HOST --device-id ID [options]

Options:
  --serial SERIAL            adb device serial when more than one phone is connected
  --coordinator-host HOST    coordinator machine LAN IP or host name
  --coordinator-port PORT    coordinator gRPC port, default 50051
  --device-id ID             deviceId matching coordinator/config/pipeline.json
  --local-port PORT          worker data-plane port, default 26052
  --install                  build and install app-debug.apk before launching
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --serial) serial="$2"; shift 2 ;;
    --coordinator-host) coordinator_host="$2"; shift 2 ;;
    --coordinator-port) coordinator_port="$2"; shift 2 ;;
    --device-id) device_id="$2"; shift 2 ;;
    --local-port) local_port="$2"; shift 2 ;;
    --install) install=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$coordinator_host" || -z "$device_id" ]]; then
  usage >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
adb_target=()
if [[ -n "$serial" ]]; then
  adb_target=(-s "$serial")
fi

if [[ "$install" == true ]]; then
  "$repo_root/gradlew" :app:assembleDebug
  adb "${adb_target[@]}" install -r "$repo_root/app/build/outputs/apk/debug/app-debug.apk"
fi

adb "${adb_target[@]}" shell am start -S \
  -n com.example.sid_trainer/.MainActivity \
  --es sid.coordinator_host "$coordinator_host" \
  --ei sid.coordinator_port "$coordinator_port" \
  --es sid.device_id "$device_id" \
  --ei sid.local_port "$local_port" \
  --ez sid.auto_start true
