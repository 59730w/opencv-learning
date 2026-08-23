$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$root = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $root "build"
$exe = Join-Path $buildDir "day51_parallel_pipeline.exe"
$library = Join-Path $buildDir "libday51_pipeline.a"
$benchmarkInput = Join-Path $buildDir "benchmark-input"
$contractOutput = Join-Path $buildDir "contract-output"
$header = Join-Path $root "include\day51\image_pipeline.hpp"
$implementation = Join-Path $root "src\image_pipeline.cpp"
$mainSource = Join-Path $root "code\day51_threads_timing.cpp"
$cmakeFile = Join-Path $root "CMakeLists.txt"

$script:passCount = 0

function Assert-True {
    param(
        [bool]$Condition,
        [string]$Message
    )

    if (-not $Condition) {
        throw "[FAIL] $Message"
    }
    $script:passCount++
    Write-Output "[PASS] $Message"
}

function Invoke-Day51 {
    param([string[]]$Arguments)

    $output = & $exe @Arguments 2>&1 | Out-String
    return [pscustomobject]@{
        ExitCode = $LASTEXITCODE
        Output = $output
    }
}

function Get-OutputHashes {
    param([string]$Directory)

    return @(
        Get-ChildItem -LiteralPath $Directory -File |
            Sort-Object Name |
            ForEach-Object {
                "$($_.Name)=$((Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash)"
            }
    )
}

Assert-True (Test-Path -LiteralPath $header -PathType Leaf) "public header exists"
Assert-True (Test-Path -LiteralPath $implementation -PathType Leaf) "parallel implementation exists"
Assert-True (Test-Path -LiteralPath $mainSource -PathType Leaf) "thin main program exists"
Assert-True (Test-Path -LiteralPath $cmakeFile -PathType Leaf) "CMake project exists"
Assert-True (Test-Path -LiteralPath $exe -PathType Leaf) "debug executable exists"
Assert-True (Test-Path -LiteralPath $library -PathType Leaf) "static pipeline library exists"

$headerText = Get-Content -LiteralPath $header -Raw
$implementationText = Get-Content -LiteralPath $implementation -Raw
$mainText = Get-Content -LiteralPath $mainSource -Raw
$cmakeText = Get-Content -LiteralPath $cmakeFile -Raw

Assert-True ($headerText -match 'enum\s+class\s+ExecutionMode') "header declares execution modes"
Assert-True ($headerText -match 'struct\s+PipelineReport') "header declares a timed report"
Assert-True ($implementationText -match 'std::thread') "implementation creates standard threads"
Assert-True ($implementationText -match 'std::atomic') "implementation uses an atomic task index"
Assert-True ($implementationText -match 'fetch_add') "workers claim unique indexes atomically"
Assert-True ($implementationText -match '\.join\s*\(') "implementation joins worker threads"
Assert-True ($implementationText -match 'steady_clock') "implementation uses a monotonic clock"
Assert-True ($implementationText -match 'cv::setNumThreads\s*\(\s*1\s*\)') "benchmark isolates outer threading from OpenCV internal threads"
Assert-True ($implementationText -match 'results\.resize') "results are sized before parallel writes"
Assert-True ($mainText -notmatch 'std::(thread|atomic)') "main does not contain threading details"
Assert-True ($cmakeText -match 'find_package\s*\(\s*Threads\s+REQUIRED') "CMake finds the thread package"
Assert-True ($cmakeText -match 'Threads::Threads') "pipeline links the CMake thread target"

& (Join-Path $PSScriptRoot "prepare_benchmark.ps1") -Count 32 | Out-Null
Assert-True ((Get-ChildItem -LiteralPath $benchmarkInput -File).Count -eq 32) "benchmark contains 32 generated images"

if (Test-Path -LiteralPath $contractOutput) {
    Remove-Item -LiteralPath $contractOutput -Recurse -Force
}

$cases = @(
    [pscustomobject]@{ Mode = "sequential"; Workers = 1 },
    [pscustomobject]@{ Mode = "threads"; Workers = 1 },
    [pscustomobject]@{ Mode = "threads"; Workers = 2 },
    [pscustomobject]@{ Mode = "threads"; Workers = 4 }
)

$baselineHashes = $null
foreach ($case in $cases) {
    $label = "$($case.Mode)-$($case.Workers)"
    $outputDir = Join-Path $contractOutput $label
    $run = Invoke-Day51 @(
        $benchmarkInput,
        $outputDir,
        "--op", "edge",
        "--mode", $case.Mode,
        "--workers", [string]$case.Workers
    )

    Assert-True ($run.ExitCode -eq 0) "$label exits with 0"
    Assert-True ($run.Output -match "Mode: $($case.Mode)") "$label reports its mode"
    Assert-True ($run.Output -match "Requested workers: $($case.Workers)") "$label reports requested workers"
    Assert-True ($run.Output -match 'Effective workers: [1-9][0-9]*') "$label reports effective workers"
    Assert-True ($run.Output -match 'Elapsed: [0-9]+(\.[0-9]+)? ms') "$label reports elapsed milliseconds"
    Assert-True ($run.Output -match 'Throughput: [0-9]+(\.[0-9]+)? images/s') "$label reports throughput"
    Assert-True ($run.Output -match 'Saved 32 / 32 image\(s\)') "$label saves every image"
    Assert-True ($run.Output -match 'DAY51_THREADS_OK') "$label prints the success marker"
    Assert-True ((Get-ChildItem -LiteralPath $outputDir -File).Count -eq 32) "$label creates 32 outputs"

    $hashes = Get-OutputHashes $outputDir
    if ($null -eq $baselineHashes) {
        $baselineHashes = $hashes
    } else {
        Assert-True (-not (Compare-Object $baselineHashes $hashes)) "$label hashes match sequential output"
    }
}

$singleImage = Join-Path $root "assets\input"
$clampedOutput = Join-Path $contractOutput "clamped"
$clamped = Invoke-Day51 @($singleImage, $clampedOutput, "--op", "gray", "--mode", "threads", "--workers", "8")
Assert-True ($clamped.ExitCode -eq 0) "worker clamping run exits with 0"
Assert-True ($clamped.Output -match 'Effective workers: 1') "workers are clamped to one input image"

$emptyInput = Join-Path $buildDir "empty-input"
New-Item -ItemType Directory -Force -Path $emptyInput | Out-Null
$emptyOutput = Join-Path $contractOutput "empty"
$empty = Invoke-Day51 @($emptyInput, $emptyOutput, "--op", "gray", "--mode", "threads", "--workers", "4")
Assert-True ($empty.ExitCode -eq 0) "empty directory exits with 0"
Assert-True ($empty.Output -match 'Files: 0') "empty directory reports zero files"
Assert-True ($empty.Output -match 'Throughput: 0\.00 images/s') "empty directory avoids division by zero"

$missingDir = Invoke-Day51 @((Join-Path $root "assets\missing"), (Join-Path $contractOutput "missing"), "--op", "gray", "--mode", "sequential", "--workers", "1")
Assert-True ($missingDir.ExitCode -eq 1) "missing input directory exits with 1"

$missingArgs = Invoke-Day51 @()
Assert-True ($missingArgs.ExitCode -eq 2) "missing arguments exit with 2"

$blockedOutput = Invoke-Day51 @($singleImage, (Join-Path $singleImage "day45_input.jpg"), "--op", "gray", "--mode", "sequential", "--workers", "1")
Assert-True ($blockedOutput.ExitCode -eq 3) "non-directory output path exits with 3"

$invalidOperation = Invoke-Day51 @($singleImage, (Join-Path $contractOutput "invalid-op"), "--op", "rotate", "--mode", "sequential", "--workers", "1")
Assert-True ($invalidOperation.ExitCode -eq 4) "unknown operation exits with 4"

$invalidMode = Invoke-Day51 @($singleImage, (Join-Path $contractOutput "invalid-mode"), "--op", "gray", "--mode", "async", "--workers", "1")
Assert-True ($invalidMode.ExitCode -eq 5) "unknown execution mode exits with 5"

foreach ($invalidWorker in @("0", "-1", "abc")) {
    $invalidWorkers = Invoke-Day51 @($singleImage, (Join-Path $contractOutput "invalid-workers"), "--op", "gray", "--mode", "threads", "--workers", $invalidWorker)
    Assert-True ($invalidWorkers.ExitCode -eq 6) "worker value '$invalidWorker' exits with 6"
}

Write-Output "Passed $script:passCount Day51 checks."
Write-Output "DAY51_CONTRACT_OK"
