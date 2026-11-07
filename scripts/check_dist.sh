#!/usr/bin/env bash
# Check built distributions before they are published: their metadata, that they carry the
# expected version, and that each installs into a fresh environment and runs.
#   scripts/check_dist.sh DIST_DIR [VERSION]    VERSION defaults to pyproject.toml's
set -euo pipefail

dist=${1:?usage: scripts/check_dist.sh DIST_DIR [VERSION]}
version=${2:-$(sed -n 's/^version = "\(.*\)"$/\1/p' pyproject.toml | head -1)}
wheel="$dist/agentloom-$version-py3-none-any.whl"
sdist="$dist/agentloom-$version.tar.gz"
for file in "$wheel" "$sdist"; do
  [ -f "$file" ] || { echo "missing $file (is $version the built version?)" >&2; exit 1; }
done

uvx --from "twine==7.0.0" twine check --strict "$wheel" "$sdist"

for file in "$wheel" "$sdist"; do
  env_dir=$(mktemp -d)
  uv venv -q "$env_dir"
  uv pip install -q --python "$env_dir/bin/python" "$file"
  info=$("$env_dir/bin/agentloom" info)  # captured first: grep -q in a pipe would SIGPIPE it
  grep -qF "AgentLoom v$version" <<<"$info" \
    || { echo "$file: agentloom info does not report $version" >&2; exit 1; }
  # Lazy imports and package data (the pricing table) only show up in an installed package
  "$env_dir/bin/python" -c "import agentloom.contracts
from agentloom.providers.pricing import DEFAULT_PRICING
assert DEFAULT_PRICING, 'empty pricing table'"
  echo "ok: $(basename "$file")"
done
echo "distributions OK: agentloom $version"
