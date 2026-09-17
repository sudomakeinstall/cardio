#!/usr/bin/env bash
# Run the suite inside a memory cap, so a test that runs away takes the run
# down with it rather than the machine.

set -euo pipefail

MEMORY_MAX="${CARDIO_TEST_MEMORY_MAX:-8G}"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="$root/.venv/bin/python"
[ -x "$python" ] || python="$(command -v python3)"

capped=0
if command -v systemd-run >/dev/null 2>&1 &&
    systemd-run --user --scope --quiet \
        --property=MemoryMax="$MEMORY_MAX" \
        --property=MemorySwapMax=0 \
        -- true >/dev/null 2>&1; then
    capped=1
else
    echo "scripts/pytest-with-memory-cap.sh: no usable cgroup cap; running uncapped" >&2
fi

status=0
if [ "$capped" -eq 1 ]; then
    systemd-run --user --scope --quiet \
        --property=MemoryMax="$MEMORY_MAX" \
        --property=MemorySwapMax=0 \
        -- "$python" -m pytest "$@" || status=$?
else
    "$python" -m pytest "$@" || status=$?
fi

if [ "$status" -eq 137 ]; then
    echo >&2
    echo "scripts/pytest-with-memory-cap.sh: the run was killed at $MEMORY_MAX." >&2
    echo "Raise it for this run with CARDIO_TEST_MEMORY_MAX, or find what grew." >&2
fi

exit "$status"
