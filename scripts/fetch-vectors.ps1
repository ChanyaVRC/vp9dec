<#
.SYNOPSIS
Downloads the official libvpx VP9 conformance vectors this project's test suite uses
(see scripts/vectors.txt) into tests/vectors/. Idempotent: skips any file that's already
present. For vectors that upstream only ships as .webm, also remuxes to .ivf via
`cargo run --example webm_to_ivf` and copies the .webm.md5 alongside it as .ivf.md5 (the
MD5s are of the decoded pixel output, not the container, so they carry over unchanged).

.EXAMPLE
pwsh scripts/fetch-vectors.ps1
#>
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$VectorsDir = Join-Path $RepoRoot "tests\vectors"
$BaseUrl = "https://storage.googleapis.com/downloads.webmproject.org/test_data/libvpx"

New-Item -ItemType Directory -Force -Path $VectorsDir | Out-Null

# Downloads $Url to $Out, skipping if $Out already exists. Invoke-WebRequest throws a
# terminating error on HTTP/network failure, and $ErrorActionPreference = "Stop" above makes
# that fail the whole script loudly rather than continuing silently.
function Get-Vector {
    param([string]$Url, [string]$Out)
    if (Test-Path $Out) {
        Write-Host "[skip] $Out already present"
        return
    }
    Write-Host "[fetch] $Url -> $Out"
    Invoke-WebRequest -Uri $Url -OutFile $Out
}

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
            Get-Vector "$BaseUrl/$name.ivf" (Join-Path $VectorsDir "$name.ivf")
            Get-Vector "$BaseUrl/$name.ivf.md5" (Join-Path $VectorsDir "$name.ivf.md5")
        }
        "webm" {
            Get-Vector "$BaseUrl/$name.webm" (Join-Path $VectorsDir "$name.webm")
            Get-Vector "$BaseUrl/$name.webm.md5" (Join-Path $VectorsDir "$name.webm.md5")

            $ivfPath = Join-Path $VectorsDir "$name.ivf"
            if (Test-Path $ivfPath) {
                Write-Host "[skip] $ivfPath already present"
            } else {
                Write-Host "[remux] $name.webm -> $name.ivf"
                Push-Location $RepoRoot
                try {
                    cargo run --example webm_to_ivf -- "tests/vectors/$name.webm" "tests/vectors/$name.ivf"
                    if ($LASTEXITCODE -ne 0) {
                        throw "webm_to_ivf failed for $name (exit code $LASTEXITCODE)"
                    }
                } finally {
                    Pop-Location
                }
            }

            $ivfMd5Path = Join-Path $VectorsDir "$name.ivf.md5"
            if (Test-Path $ivfMd5Path) {
                Write-Host "[skip] $ivfMd5Path already present"
            } else {
                Copy-Item (Join-Path $VectorsDir "$name.webm.md5") $ivfMd5Path
            }
        }
        default {
            throw "unknown vector kind '$kind' for '$name' in scripts/vectors.txt"
        }
    }
}

Write-Host "[done] all vectors present in $VectorsDir"
