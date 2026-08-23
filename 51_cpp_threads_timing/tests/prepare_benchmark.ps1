param(
    [ValidateRange(1, 1000)]
    [int]$Count = 32
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$sourceImage = Join-Path $root "assets\input\day45_input.jpg"
$benchmarkDir = Join-Path $root "build\benchmark-input"

if (-not (Test-Path -LiteralPath $sourceImage -PathType Leaf)) {
    throw "Source image does not exist: $sourceImage"
}

if (Test-Path -LiteralPath $benchmarkDir) {
    Remove-Item -LiteralPath $benchmarkDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $benchmarkDir | Out-Null

for ($index = 1; $index -le $Count; $index++) {
    $name = "tree_{0:D2}.jpg" -f $index
    Copy-Item -LiteralPath $sourceImage -Destination (Join-Path $benchmarkDir $name)
}

$actualCount = (Get-ChildItem -LiteralPath $benchmarkDir -File).Count
if ($actualCount -ne $Count) {
    throw "Expected $Count benchmark images, found $actualCount."
}

Write-Output "Prepared $actualCount benchmark images in $benchmarkDir"
Write-Output "DAY51_BENCHMARK_INPUT_OK"
