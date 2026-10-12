#!/usr/bin/env bash
set -euo pipefail

if (($# < 1 || $# > 2)); then
    printf 'Usage: %s PATTERN [SINCE]\n' "$0" >&2
    printf 'Example: %s 4459019648 "7 days ago"\n' "$0" >&2
    exit 2
fi

pattern="$1"
since="${2:-7 days ago}"

mapfile -t container_names < <(docker compose ps -a --format '{{.Name}}' 2>/dev/null | sort -u)
if ((${#container_names[@]} == 0)); then
    printf 'No Compose containers found.\n' >&2
    exit 1
fi

found=false
for container_name in "${container_names[@]}"; do
    while IFS= read -r line; do
        if grep -Fq -- "$pattern" <<<"$line"; then
            printf '[%s] %s\n' "$container_name" "$line"
            found=true
        fi
    done < <(
        journalctl \
            --since "$since" \
            --no-pager \
            --output=short-iso \
            "CONTAINER_NAME=$container_name" 2>/dev/null || true
    )
done

if [[ "$found" != true ]]; then
    printf 'No persisted container logs matched %q since %q.\n' "$pattern" "$since" >&2
    exit 1
fi
