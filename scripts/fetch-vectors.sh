#!/usr/bin/env bash
# Downloads the official libvpx VP9 conformance vectors this project's test suite uses
# (see scripts/vectors.txt) into tests/vectors/. Idempotent: skips any file that's already
# present. For vectors that upstream only ships as .webm, also remuxes to .ivf via
# `cargo run --example webm_to_ivf` and copies the .webm.md5 alongside it as .ivf.md5 (the
# MD5s are of the decoded pixel output, not the container, so they carry over unchanged).
#
# Usage: bash scripts/fetch-vectors.sh
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
vectors_dir="$repo_root/tests/vectors"
base_url="https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx"

mkdir -p "$vectors_dir"

# Downloads $1 to $2, skipping if $2 already exists. Fails loudly (via `set -e` + curl -f) on
# any HTTP error or network failure.
fetch() {
    local url="$1" out="$2"
    if [ -f "$out" ]; then
        echo "[skip] $out already present"
        return
    fi
    echo "[fetch] $url -> $out"
    curl -fSL -o "$out" "$url"
}

while read -r name kind; do
    case "$name" in
        ""|"#"*) continue ;;
    esac

    case "$kind" in
        ivf)
            fetch "$base_url/$name.ivf" "$vectors_dir/$name.ivf"
            fetch "$base_url/$name.ivf.md5" "$vectors_dir/$name.ivf.md5"
            ;;
        webm)
            fetch "$base_url/$name.webm" "$vectors_dir/$name.webm"
            fetch "$base_url/$name.webm.md5" "$vectors_dir/$name.webm.md5"

            if [ -f "$vectors_dir/$name.ivf" ]; then
                echo "[skip] $vectors_dir/$name.ivf already present"
            else
                echo "[remux] $name.webm -> $name.ivf"
                (cd "$repo_root" && cargo run --example webm_to_ivf -- \
                    "tests/vectors/$name.webm" "tests/vectors/$name.ivf")
            fi

            if [ -f "$vectors_dir/$name.ivf.md5" ]; then
                echo "[skip] $vectors_dir/$name.ivf.md5 already present"
            else
                cp "$vectors_dir/$name.webm.md5" "$vectors_dir/$name.ivf.md5"
            fi
            ;;
        *)
            echo "[error] unknown vector kind '$kind' for '$name' in scripts/vectors.txt" >&2
            exit 1
            ;;
    esac
done < "$script_dir/vectors.txt"

echo "[done] all vectors present in $vectors_dir"
