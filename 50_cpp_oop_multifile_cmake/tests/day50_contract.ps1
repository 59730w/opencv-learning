$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$root = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $root "build"
$exe = Join-Path $buildDir "day50_oop_pipeline.exe"
$library = Join-Path $buildDir "libday50_pipeline.a"
$inputDir = Join-Path $root "assets\input"
$contractOutput = Join-Path $buildDir "contract-output"
$header = Join-Path $root "include\day50\image_pipeline.hpp"
$implementation = Join-Path $root "src\image_pipeline.cpp"
$mainSource = Join-Path $root "code\day50_oop_multifile.cpp"
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

function Invoke-Day50 {
    param([string[]]$Arguments)

    $output = & $exe @Arguments 2>&1 | Out-String
    return [pscustomobject]@{
        ExitCode = $LASTEXITCODE
        Output = $output
    }
}

Assert-True (Test-Path -LiteralPath $header -PathType Leaf) "public header exists"
Assert-True (Test-Path -LiteralPath $implementation -PathType Leaf) "class implementation exists"
Assert-True (Test-Path -LiteralPath $mainSource -PathType Leaf) "thin main program exists"
Assert-True (Test-Path -LiteralPath $cmakeFile -PathType Leaf) "CMake project exists"
Assert-True (Test-Path -LiteralPath $exe -PathType Leaf) "debug executable exists"
Assert-True (Test-Path -LiteralPath $library -PathType Leaf) "static pipeline library exists"

$headerText = Get-Content -LiteralPath $header -Raw
$implementationText = Get-Content -LiteralPath $implementation -Raw
$mainText = Get-Content -LiteralPath $mainSource -Raw
$cmakeText = Get-Content -LiteralPath $cmakeFile -Raw

Assert-True ($headerText -match 'class\s+ImagePipeline') "header declares ImagePipeline"
Assert-True ($headerText -match 'public\s*:') "class exposes a public interface"
Assert-True ($headerText -match 'private\s*:') "class hides implementation details"
Assert-True ($implementationText -match 'ImagePipeline::ImagePipeline') "constructor is defined out of line"
Assert-True ($mainText -match 'ImagePipeline\s+pipeline') "main creates a pipeline object"
Assert-True ($mainText -notmatch 'cv::(cvtColor|GaussianBlur|Canny|imwrite)') "main does not contain processing details"
Assert-True ($cmakeText -match 'add_library\s*\(\s*day50_pipeline\s+STATIC') "CMake creates a static library target"
Assert-True ($cmakeText -match 'add_executable\s*\(\s*day50_oop_pipeline') "CMake creates an executable target"
Assert-True ($cmakeText -match 'target_link_libraries\s*\(\s*day50_oop_pipeline[\s\S]*day50_pipeline') "executable links the pipeline library"

if (Test-Path -LiteralPath $contractOutput) {
    Remove-Item -LiteralPath $contractOutput -Recurse -Force
}

$hashes = @()
foreach ($operation in @("gray", "blur", "edge")) {
    $outputDir = Join-Path $contractOutput $operation
    $run = Invoke-Day50 @($inputDir, $outputDir, "--op", $operation)
    $outputImage = Join-Path $outputDir "day45_input_${operation}.jpg"

    Assert-True ($run.ExitCode -eq 0) "$operation mode exits with 0"
    Assert-True ($run.Output -match "Operation: $operation") "$operation mode reports its operation"
    Assert-True ($run.Output -match 'Saved 1 / 1 image\(s\)') "$operation mode saves the input image"
    Assert-True ($run.Output -match 'DAY50_OOP_OK') "$operation mode prints the success marker"
    Assert-True (Test-Path -LiteralPath $outputImage -PathType Leaf) "$operation output image exists"
    $hashes += (Get-FileHash -LiteralPath $outputImage -Algorithm SHA256).Hash
}

Assert-True (($hashes | Select-Object -Unique).Count -eq 3) "gray, blur, and edge outputs are distinct"

$missingDir = Invoke-Day50 @((Join-Path $root "assets\missing"), (Join-Path $contractOutput "missing"), "--op", "gray")
Assert-True ($missingDir.ExitCode -eq 1) "missing input directory exits with 1"

$missingArgs = Invoke-Day50 @()
Assert-True ($missingArgs.ExitCode -eq 2) "missing arguments exit with 2"

$blockedOutput = Invoke-Day50 @($inputDir, (Join-Path $inputDir "day45_input.jpg"), "--op", "gray")
Assert-True ($blockedOutput.ExitCode -eq 3) "non-directory output path exits with 3"

$invalidOperation = Invoke-Day50 @($inputDir, (Join-Path $contractOutput "invalid"), "--op", "rotate")
Assert-True ($invalidOperation.ExitCode -eq 4) "unknown operation exits with 4"

Write-Output "Passed $script:passCount Day50 checks."
Write-Output "DAY50_CONTRACT_OK"
