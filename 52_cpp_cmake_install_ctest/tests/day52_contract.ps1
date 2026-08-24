param(
    [Parameter(Mandatory = $true)]
    [string]$Executable
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$root = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $root "build"
$assets = Join-Path $root "assets\input"
$contractRoot = Join-Path $buildDir "contract-output"
$header = Join-Path $root "include\day52\image_pipeline.hpp"
$implementation = Join-Path $root "src\image_pipeline.cpp"
$mainSource = Join-Path $root "code\day52_cmake_install_ctest.cpp"
$unitSource = Join-Path $root "tests\day52_unit_tests.cpp"
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

function Invoke-Day52 {
    param([string[]]$Arguments)

    $output = & $Executable @Arguments 2>&1 | Out-String
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

Assert-True (Test-Path -LiteralPath $Executable -PathType Leaf) "debug executable exists"
Assert-True (Test-Path -LiteralPath $header -PathType Leaf) "public header exists"
Assert-True (Test-Path -LiteralPath $implementation -PathType Leaf) "pipeline implementation exists"
Assert-True (Test-Path -LiteralPath $mainSource -PathType Leaf) "thin main program exists"
Assert-True (Test-Path -LiteralPath $unitSource -PathType Leaf) "C++ unit-test source exists"
Assert-True (Test-Path -LiteralPath $cmakeFile -PathType Leaf) "CMake project exists"
Assert-True ((Get-ChildItem -LiteralPath $assets -File).Count -eq 4) "four representative input images exist"

$headerText = Get-Content -LiteralPath $header -Raw
$implementationText = Get-Content -LiteralPath $implementation -Raw
$mainText = Get-Content -LiteralPath $mainSource -Raw
$unitText = Get-Content -LiteralPath $unitSource -Raw
$cmakeText = Get-Content -LiteralPath $cmakeFile -Raw

Assert-True ($headerText -match 'resolve_worker_count') "worker policy is exposed as a testable function"
Assert-True ($implementationText -match 'std::thread') "pipeline retains Day51 standard threads"
Assert-True ($implementationText -match 'cv::setNumThreads\s*\(\s*1\s*\)') "OpenCV nested parallelism remains controlled"
Assert-True ($mainText -notmatch 'std::(thread|atomic)') "main remains free of threading details"
Assert-True ($unitText -match 'DAY52_UNIT_TESTS_OK') "unit test has a deterministic marker"
Assert-True ($cmakeText -match 'include\s*\(\s*CTest\s*\)') "CMake enables CTest"
Assert-True ($cmakeText -match 'add_test\s*\(') "CMake registers tests"
Assert-True ($cmakeText -match '\$<BUILD_INTERFACE:') "build-tree include interface is declared"
Assert-True ($cmakeText -match '\$<INSTALL_INTERFACE:') "install-tree include interface is declared"
Assert-True ($cmakeText -match 'install\s*\(\s*EXPORT') "CMake exports installed targets"
Assert-True ($cmakeText -match 'configure_package_config_file') "CMake creates a relocatable package config"
Assert-True ($cmakeText -match 'include\s*\(\s*CPack\s*\)') "CMake enables CPack"

if (Test-Path -LiteralPath $contractRoot) {
    Remove-Item -LiteralPath $contractRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $contractRoot | Out-Null

$operationRuns = @{}
foreach ($operation in @("gray", "blur", "edge")) {
    $outputDir = Join-Path $contractRoot "sequential-$operation"
    $run = Invoke-Day52 @(
        $assets,
        $outputDir,
        "--op", $operation,
        "--mode", "sequential",
        "--workers", "8"
    )
    $operationRuns[$operation] = $outputDir
    Assert-True ($run.ExitCode -eq 0) "$operation sequential run exits with 0"
    Assert-True ($run.Output -match "Operation: $operation") "$operation run reports its operation"
    Assert-True ($run.Output -match 'Effective workers: 1') "$operation sequential run uses one effective worker"
    Assert-True ($run.Output -match 'Saved 4 / 4 image\(s\)') "$operation run saves all four images"
    Assert-True ($run.Output -match 'DAY52_PIPELINE_OK') "$operation run prints the success marker"
    Assert-True ((Get-ChildItem -LiteralPath $outputDir -File).Count -eq 4) "$operation run creates four files"
}

$threadedOutput = Join-Path $contractRoot "threads-edge"
$threaded = Invoke-Day52 @(
    $assets,
    $threadedOutput,
    "--op", "edge",
    "--mode", "threads",
    "--workers", "8"
)
Assert-True ($threaded.ExitCode -eq 0) "threaded edge run exits with 0"
Assert-True ($threaded.Output -match 'Mode: threads') "threaded edge run reports threads mode"
Assert-True ($threaded.Output -match 'Requested workers: 8') "threaded run reports requested workers"
Assert-True ($threaded.Output -match 'Effective workers: 4') "workers are capped by four images"
Assert-True ($threaded.Output -match 'Saved 4 / 4 image\(s\)') "threaded run saves all images"
Assert-True ($threaded.Output -match 'DAY52_PIPELINE_OK') "threaded run prints the success marker"
Assert-True (-not (Compare-Object (Get-OutputHashes $operationRuns["edge"]) (Get-OutputHashes $threadedOutput))) "threaded hashes match sequential hashes"

$rereadOutput = Join-Path $contractRoot "reread"
$reread = Invoke-Day52 @(
    $operationRuns["blur"],
    $rereadOutput,
    "--op", "gray",
    "--mode", "sequential",
    "--workers", "1"
)
Assert-True ($reread.ExitCode -eq 0) "generated JPEG files can be read again"
Assert-True ($reread.Output -match 'Saved 4 / 4 image\(s\)') "re-read run saves all outputs"

$corruptInput = Join-Path $buildDir "corrupt-input"
New-Item -ItemType Directory -Force -Path $corruptInput | Out-Null
Get-ChildItem -LiteralPath $assets -File |
    Copy-Item -Destination $corruptInput -Force
Set-Content -LiteralPath (Join-Path $corruptInput "corrupt.jpg") -Value "not an image" -Encoding ASCII
$corruptOutput = Join-Path $contractRoot "corrupt"
$corrupt = Invoke-Day52 @(
    $corruptInput,
    $corruptOutput,
    "--op", "gray",
    "--mode", "threads",
    "--workers", "4"
)
Assert-True ($corrupt.ExitCode -eq 0) "one unreadable image does not abort the batch"
Assert-True ($corrupt.Output -match 'Saved 4 / 5 image\(s\); read-failed 1') "unreadable image is isolated and reported"
Assert-True ((Get-ChildItem -LiteralPath $corruptOutput -File).Count -eq 4) "valid neighbors still produce outputs"

$emptyInput = Join-Path $buildDir "empty-input"
New-Item -ItemType Directory -Force -Path $emptyInput | Out-Null
$empty = Invoke-Day52 @(
    $emptyInput,
    (Join-Path $contractRoot "empty"),
    "--op", "gray",
    "--mode", "threads",
    "--workers", "4"
)
Assert-True ($empty.ExitCode -eq 0) "empty directory exits with 0"
Assert-True ($empty.Output -match 'Files: 0') "empty directory reports zero files"
Assert-True ($empty.Output -match 'Throughput: 0\.00 images/s') "empty directory avoids division by zero"

$missingDir = Invoke-Day52 @((Join-Path $root "assets\missing"), (Join-Path $contractRoot "missing"), "--op", "gray", "--mode", "sequential", "--workers", "1")
Assert-True ($missingDir.ExitCode -eq 1) "missing input directory exits with 1"
$missingArgs = Invoke-Day52 @()
Assert-True ($missingArgs.ExitCode -eq 2) "missing arguments exit with 2"
$blockedOutput = Invoke-Day52 @($assets, (Join-Path $assets "personal_563x845.jpg"), "--op", "gray", "--mode", "sequential", "--workers", "1")
Assert-True ($blockedOutput.ExitCode -eq 3) "non-directory output path exits with 3"
$invalidOperation = Invoke-Day52 @($assets, (Join-Path $contractRoot "invalid-op"), "--op", "rotate", "--mode", "sequential", "--workers", "1")
Assert-True ($invalidOperation.ExitCode -eq 4) "unknown operation exits with 4"
$invalidMode = Invoke-Day52 @($assets, (Join-Path $contractRoot "invalid-mode"), "--op", "gray", "--mode", "async", "--workers", "1")
Assert-True ($invalidMode.ExitCode -eq 5) "unknown execution mode exits with 5"
foreach ($invalidWorker in @("0", "-1", "abc")) {
    $invalidWorkers = Invoke-Day52 @($assets, (Join-Path $contractRoot "invalid-workers"), "--op", "gray", "--mode", "threads", "--workers", $invalidWorker)
    Assert-True ($invalidWorkers.ExitCode -eq 6) "worker value '$invalidWorker' exits with 6"
}

Write-Output "Passed $script:passCount Day52 contract checks."
Write-Output "DAY52_CONTRACT_OK"
