<#
.SYNOPSIS
Downloads the official libvpx VP9 conformance vectors this project's test suite uses
(see scripts/vectors.txt) into tests/vectors/. Idempotent: skips any file that's already
present. For vectors that upstream only ships as .webm, also remuxes to .ivf via
`cargo run --example webm_to_ivf` and copies the .webm.md5 alongside it as .ivf.md5 (the
MD5s are of the decoded pixel output, not the container, so they carry over unchanged).

Individual download/remux failures are recorded and skipped rather than aborting the whole
run (M4 wave 1): at ~330 manifest entries, some upstream files 404 (a vector may ship only
one container form, or lack a .md5) and some .webm files may not remux cleanly -- both are
expected data about the vector set, not reasons to stop fetching the rest. A summary with
every failing name is printed at the end; see docs/implementation-notes.md "M4 wave 1".

.EXAMPLE
pwsh scripts/fetch-vectors.ps1
#>

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$VectorsDir = Join-Path $RepoRoot "tests\vectors"
$BaseUrl = "https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx"

New-Item -ItemType Directory -Force -Path $VectorsDir | Out-Null

$DlOk = [System.Collections.Generic.List[string]]::new()
$DlFail = [System.Collections.Generic.List[string]]::new()
$RemuxOk = [System.Collections.Generic.List[string]]::new()
$RemuxFail = [System.Collections.Generic.List[string]]::new()

# Downloads $Url to $Out, skipping if $Out already exists. Records the outcome in
# $DlOk/$DlFail instead of letting a single HTTP/network failure abort the whole run, and
# returns whether it succeeded so callers can react (e.g. skip a remux whose source
# download failed).
function Get-Vector {
    param([string]$Url, [string]$Out)
    if (Test-Path $Out) {
        Write-Host "[skip] $Out already present"
        return $true
    }
    Write-Host "[fetch] $Url -> $Out"
    try {
        Invoke-WebRequest -Uri $Url -OutFile $Out -ErrorAction Stop
        $DlOk.Add($Out)
        return $true
    } catch {
        Write-Warning "[FAIL] download failed: $Url ($($_.Exception.Message))"
        Remove-Item -Force -ErrorAction SilentlyContinue $Out
        $DlFail.Add($Url)
        return $false
    }
}

# Remuxes tests/vectors/$Src (a .webm) to tests/vectors/$Dst (an .ivf) via the pre-built
# webm_to_ivf example, recording the outcome in $RemuxOk/$RemuxFail.
function Invoke-Remux {
    param([string]$Src, [string]$Dst)
    Write-Host "[remux] $Src -> $Dst"
    Push-Location $RepoRoot
    try {
        $remuxErr = cargo run --quiet --example webm_to_ivf -- "tests/vectors/$Src" "tests/vectors/$Dst" 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "[FAIL] remux failed: $Src"
            $remuxErr | ForEach-Object { Write-Warning $_ }
            $RemuxFail.Add("$Src`: $($remuxErr | Select-Object -Last 1)")
        } else {
            $RemuxOk.Add($Src)
        }
    } finally {
        Pop-Location
    }
}

# Pre-build once so the ~300 subsequent `cargo run --example webm_to_ivf` invocations below
# each skip straight to executing the (already up to date) binary.
cargo build --quiet --example webm_to_ivf

Get-Content (Join-Path $ScriptDir "vectors.txt") | ForEach-Object {
    $line = $_.Trim()
    if ($line -eq "" -or $line.StartsWith("#")) {
        return
    }
    $parts = $line -split '\s+'
    $name = $parts[0]
    $kind = $parts[1]

    switch ($kind) {
        "ivf" {
            Get-Vector "$BaseUrl/$name.ivf" (Join-Path $VectorsDir "$name.ivf") | Out-Null
            Get-Vector "$BaseUrl/$name.ivf.md5" (Join-Path $VectorsDir "$name.ivf.md5") | Out-Null
        }
        "webm" {
            $webmOk = Get-Vector "$BaseUrl/$name.webm" (Join-Path $VectorsDir "$name.webm")
            Get-Vector "$BaseUrl/$name.webm.md5" (Join-Path $VectorsDir "$name.webm.md5") | Out-Null

            $ivfPath = Join-Path $VectorsDir "$name.ivf"
            $webmPath = Join-Path $VectorsDir "$name.webm"
            if (Test-Path $ivfPath) {
                Write-Host "[skip] $ivfPath already present"
            } elseif (-not $webmOk -and -not (Test-Path $webmPath)) {
                Write-Host "[skip] $name.webm not available, cannot remux"
            } else {
                Invoke-Remux "$name.webm" "$name.ivf"
            }

            $ivfMd5Path = Join-Path $VectorsDir "$name.ivf.md5"
            $webmMd5Path = Join-Path $VectorsDir "$name.webm.md5"
            if (Test-Path $ivfMd5Path) {
                Write-Host "[skip] $ivfMd5Path already present"
            } elseif (Test-Path $webmMd5Path) {
                Copy-Item $webmMd5Path $ivfMd5Path
            }
        }
        "invalid" {
            # `name` is the full upstream filename; download it and its `.res` sidecar verbatim.
            $ok = Get-Vector "$BaseUrl/$name" (Join-Path $VectorsDir $name)
            Get-Vector "$BaseUrl/$name.res" (Join-Path $VectorsDir "$name.res") | Out-Null

            # A .webm invalid vector is remuxed to <name>.ivf so the IVF-based test can read it;
            # its .res (per-decoded-frame codes, container-agnostic) is copied to <name>.ivf.res.
            if ($name -like "*.webm") {
                $ivfPath = Join-Path $VectorsDir "$name.ivf"
                $webmPath = Join-Path $VectorsDir $name
                if (Test-Path $ivfPath) {
                    Write-Host "[skip] $ivfPath already present"
                } elseif (-not $ok -and -not (Test-Path $webmPath)) {
                    Write-Host "[skip] $name not available, cannot remux"
                } else {
                    Invoke-Remux $name "$name.ivf"
                }

                $ivfResPath = Join-Path $VectorsDir "$name.ivf.res"
                $webmResPath = Join-Path $VectorsDir "$name.res"
                if (Test-Path $ivfResPath) {
                    Write-Host "[skip] $ivfResPath already present"
                } elseif (Test-Path $webmResPath) {
                    Copy-Item $webmResPath $ivfResPath
                }
            }
        }
        default {
            Write-Warning "[error] unknown vector kind '$kind' for '$name' in scripts/vectors.txt"
        }
    }
}

Write-Host ""
Write-Host "===== fetch-vectors.ps1 summary ====="
Write-Host "downloads ok:     $($DlOk.Count)"
Write-Host "downloads failed: $($DlFail.Count)"
$DlFail | ForEach-Object { Write-Host "  [download-fail] $_" }
Write-Host "remux ok:         $($RemuxOk.Count)"
Write-Host "remux failed:     $($RemuxFail.Count)"
$RemuxFail | ForEach-Object { Write-Host "  [remux-fail] $_" }
Write-Host "======================================"
Write-Host "[done] fetch-vectors.ps1 finished (see summary above for anything that didn't succeed)"
