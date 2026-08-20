$ErrorActionPreference = 'Stop'

$day48Root = Split-Path -Parent $PSScriptRoot
$smartPointersExe = Join-Path $day48Root 'build\day48_smart_pointers.exe'
$pipelineExe = Join-Path $day48Root 'build\day48_pipeline.exe'
$assetsDir = Join-Path $day48Root 'assets'
$contractOutput = Join-Path $day48Root 'output\contract'
$failures = 0

function Assert-Day48 {
    param(
        [bool]$Condition,
        [string]$Message
    )

    if ($Condition) {
        Write-Host "[PASS] $Message"
    }
    else {
        Write-Host "[FAIL] $Message"
        $script:failures++
    }
}

Assert-Day48 (Test-Path -LiteralPath $smartPointersExe -PathType Leaf) `
    'smart-pointer executable exists'
Assert-Day48 (Test-Path -LiteralPath $pipelineExe -PathType Leaf) `
    'pipeline executable exists'

if ($failures -gt 0) {
    Write-Host "DAY48_CONTRACT_FAILED failures=$failures"
    exit 1
}

if (Test-Path -LiteralPath $contractOutput) {
    $resolvedOutput = (Resolve-Path -LiteralPath $contractOutput).Path
    if (-not $resolvedOutput.StartsWith($day48Root, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove unexpected path: $resolvedOutput"
    }
    Remove-Item -LiteralPath $resolvedOutput -Recurse -Force
}

$smartOutput = (& $smartPointersExe 2>&1 | Out-String)
$smartExit = $LASTEXITCODE
Assert-Day48 ($smartExit -eq 0) 'smart-pointer demo exits with 0'
Assert-Day48 ($smartOutput -match 'source empty: true') 'unique_ptr source becomes empty'
Assert-Day48 ($smartOutput -match 'count: 1[\s\S]*count: 2[\s\S]*count: 1') `
    'shared_ptr count changes 1 -> 2 -> 1'
Assert-Day48 ($smartOutput -match 'DAY48_SMART_POINTERS_OK') `
    'smart-pointer success marker is present'

$expectedTypes = @{ gray = 0; blur = 64; edge = 0 }
foreach ($operation in @('gray', 'blur', 'edge')) {
    $operationOutput = Join-Path $contractOutput $operation
    $pipelineOutput = (& $pipelineExe $assetsDir $operationOutput --op $operation 2>&1 | Out-String)
    $pipelineExit = $LASTEXITCODE
    $expectedImage = Join-Path $operationOutput "day45_input_$operation.jpg"

    Assert-Day48 ($pipelineExit -eq 0) "$operation exits with 0"
    Assert-Day48 ($pipelineOutput -match 'Saved 1 / 2 image\(s\); read-failed 1; write-failed 0\.') `
        "$operation isolates the broken image"
    Assert-Day48 ($pipelineOutput -match "type=$($expectedTypes[$operation])") `
        "$operation reports the expected cv::Mat type"
    Assert-Day48 ($pipelineOutput -match 'DAY48_PIPELINE_OK') `
        "$operation success marker is present"
    Assert-Day48 (Test-Path -LiteralPath $expectedImage -PathType Leaf) `
        "$operation output image exists"
}

$null = & $pipelineExe $assetsDir (Join-Path $contractOutput 'bad-op') --op sharpen 2>&1
Assert-Day48 ($LASTEXITCODE -eq 4) 'unknown operation exits with 4'

$null = & $pipelineExe 2>&1
Assert-Day48 ($LASTEXITCODE -eq 2) 'missing arguments exit with 2'

$null = & $pipelineExe (Join-Path $day48Root 'missing') `
    (Join-Path $contractOutput 'missing-input') --op gray 2>&1
Assert-Day48 ($LASTEXITCODE -eq 1) 'missing input directory exits with 1'

New-Item -ItemType Directory -Path $contractOutput -Force | Out-Null
$blockedOutput = Join-Path $contractOutput 'blocked-output'
Set-Content -LiteralPath $blockedOutput -Value 'not a directory'
$null = & $pipelineExe $assetsDir $blockedOutput --op gray 2>&1
Assert-Day48 ($LASTEXITCODE -eq 3) 'output directory creation failure exits with 3'

if ($failures -gt 0) {
    Write-Host "DAY48_CONTRACT_FAILED failures=$failures"
    exit 1
}

Write-Host 'DAY48_CONTRACT_OK'
exit 0
