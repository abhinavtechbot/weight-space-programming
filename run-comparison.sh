#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
infisical_project="${INFISICAL_PROJECT_DIR:-$HOME/Developer/adk-course-dev}"
infisical_cli="${INFISICAL_CLI:-$infisical_project/.tools/infisical/node_modules/.bin/infisical}"
if ! command -v cargo >/dev/null 2>&1; then
  export CARGO_HOME="${CARGO_HOME:-/private/tmp/weight-space-cargo}"
  export RUSTUP_HOME="${RUSTUP_HOME:-/private/tmp/weight-space-rustup}"
  export PATH="$CARGO_HOME/bin:$PATH"
fi
cargo build --release --manifest-path "$repo_dir/Cargo.toml" --bin route-demo
if command -v infisical >/dev/null 2>&1; then infisical_cli="$(command -v infisical)"; fi
if [[ ! -x "$infisical_cli" || ! -f "$infisical_project/.infisical.yaml" ]]; then
  echo "Infisical CLI or project config missing. Set INFISICAL_CLI and INFISICAL_PROJECT_DIR." >&2
  exit 1
fi
cd "$infisical_project"
if [[ -z "${INFISICAL_TOKEN:-}" ]] && command -v security >/dev/null 2>&1; then
  INFISICAL_TOKEN="$(security find-generic-password -a weight-space-demo -s weight-space-infisical -w 2>/dev/null || true)"
  export INFISICAL_TOKEN
fi
exec "$infisical_cli" run --env=dev -- python3 "$repo_dir/demo/server.py"
