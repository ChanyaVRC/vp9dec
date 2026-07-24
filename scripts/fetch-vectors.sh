#!/usr/bin/env bash
# Downloads the official libvpx VP9 conformance vectors this project's test suite uses
# (see scripts/vectors.txt) into tests/vectors/. Idempotent: skips any file that's already
# present. For vectors that upstream only ships as .webm, also remuxes to .ivf via
# `cargo run --example webm_to_ivf` and copies the .webm.md5 alongside it as .ivf.md5 (the
# MD5s are of the decoded pixel output, not the container, so they carry over unchanged).
#
# Individual download/remux failures are recorded and skipped rather than aborting the whole
# run (M4 wave 1): at ~330 manifest entries, some upstream files 404 (a vector may ship only
# one container form, or lack a .md5) and some .webm files may not remux cleanly -- both are
# expected data about the vector set, not reasons to stop fetching the rest. A summary with
# every failing name is printed at the end; see docs/implementation-notes.md "M4 wave 1".
#
# Usage: bash scripts/fetch-vectors.sh
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
vectors_dir="$repo_root/tests/vectors"
base_url="https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx"

mkdir -p "$vectors_dir"

dl_ok=()
dl_fail=()
remux_ok=()
remux_fail=()

# Downloads $1 to $2, skipping if $2 already exists. Records the outcome in dl_ok/dl_fail
# instead of letting a single HTTP/network failure abort the whole run, and returns non-zero
# so callers can react (e.g. skip a remux whose source download failed).
fetch() {
    local url="$1" out="$2"
    if [ -f "$out" ]; then
        echo "[skip] $out already present"
        return 0
    fi
    echo "[fetch] $url -> $out"
    if curl -fSL -o "$out" "$url"; then
        dl_ok+=("$out")
        return 0
    else
        local rc=$? # curl's status: after `fi` it would be the if's status (0), not curl's
        echo "[FAIL] download failed (curl exit $rc): $url" >&2
        rm -f "$out" # curl -o may leave a truncated/empty file behind on failure
        dl_fail+=("$url")
        return 1
    fi
}

# Remuxes tests/vectors/$1 (a .webm) to tests/vectors/$2 (an .ivf) via the pre-built
# webm_to_ivf example, recording the outcome in remux_ok/remux_fail.
remux_one() {
    local src="$1" dst="$2"
    echo "[remux] $src -> $dst"
    local remux_err
    remux_err="$(mktemp)"
    if (cd "$repo_root" && cargo run --quiet --example webm_to_ivf -- \
        "tests/vectors/$src" "tests/vectors/$dst") 2>"$remux_err"; then
        remux_ok+=("$src")
    else
        echo "[FAIL] remux failed: $src" >&2
        cat "$remux_err" >&2
        remux_fail+=("$src: $(tail -n 1 "$remux_err")")
    fi
    rm -f "$remux_err"
}

# Pre-build once so the ~300 subsequent `cargo run --example webm_to_ivf` invocations below
# each skip straight to executing the (already up to date) binary.
cargo build --quiet --example webm_to_ivf

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
            webm_ready=1
            fetch "$base_url/$name.webm" "$vectors_dir/$name.webm" || webm_ready=0
            fetch "$base_url/$name.webm.md5" "$vectors_dir/$name.webm.md5"

            if [ -f "$vectors_dir/$name.ivf" ]; then
                echo "[skip] $vectors_dir/$name.ivf already present"
            elif [ "$webm_ready" -eq 0 ] && [ ! -f "$vectors_dir/$name.webm" ]; then
                echo "[skip] $name.webm not available, cannot remux"
            else
                remux_one "$name.webm" "$name.ivf"
            fi

            if [ -f "$vectors_dir/$name.ivf.md5" ]; then
                echo "[skip] $vectors_dir/$name.ivf.md5 already present"
            elif [ -f "$vectors_dir/$name.webm.md5" ]; then
                cp "$vectors_dir/$name.webm.md5" "$vectors_dir/$name.ivf.md5"
            fi
            ;;
        invalid)
            # $name is the full upstream filename; download it and its .res sidecar verbatim.
            fetch "$base_url/$name" "$vectors_dir/$name" || true
            fetch "$base_url/$name.res" "$vectors_dir/$name.res"

            # A .webm invalid vector is remuxed to <name>.ivf so the IVF-based test can read it;
            # its .res (per-decoded-frame codes, container-agnostic) is copied to <name>.ivf.res.
            case "$name" in
                *.webm)
                    if [ -f "$vectors_dir/$name.ivf" ]; then
                        echo "[skip] $vectors_dir/$name.ivf already present"
                    elif [ ! -f "$vectors_dir/$name" ]; then
                        echo "[skip] $name not available, cannot remux"
                    else
                        remux_one "$name" "$name.ivf"
                    fi

                    if [ -f "$vectors_dir/$name.ivf.res" ]; then
                        echo "[skip] $vectors_dir/$name.ivf.res already present"
                    elif [ -f "$vectors_dir/$name.res" ]; then
                        cp "$vectors_dir/$name.res" "$vectors_dir/$name.ivf.res"
                    fi
                    ;;
            esac
            ;;
        *)
            echo "[error] unknown vector kind '$kind' for '$name' in scripts/vectors.txt" >&2
            ;;
    esac
done < "$script_dir/vectors.txt"

echo
echo "===== fetch-vectors.sh summary ====="
echo "downloads ok:     ${#dl_ok[@]}"
echo "downloads failed: ${#dl_fail[@]}"
for f in "${dl_fail[@]+"${dl_fail[@]}"}"; do
    echo "  [download-fail] $f"
done
echo "remux ok:         ${#remux_ok[@]}"
echo "remux failed:     ${#remux_fail[@]}"
for f in "${remux_fail[@]+"${remux_fail[@]}"}"; do
    echo "  [remux-fail] $f"
done
echo "====================================="
echo "[done] fetch-vectors.sh finished (see summary above for anything that didn't succeed)"
