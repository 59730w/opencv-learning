param(
    [ValidateRange(1, 20)]
    [int]$MeasuredRuns = 3
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$root = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $root "build"
$exe = Join-Path $buildDir "day51_parallel_pipeline.exe"
$benchmarkInput = Join-Path $buildDir "benchmark-input"
$benchmarkRoot = Join-Path $buildDir "benchmark-runs"
$rawCsv = Join-Path $buildDir "day51_benchmark_raw.csv"
$summaryCsv = Join-Path $buildDir "day51_benchmark_summary.csv"

if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
    throw "Executable does not exist: $exe"
}

& (Join-Path $PSScriptRoot "prepare_benchmark.ps1") -Count 32 | Out-Null

if (Test-Path -LiteralPath $benchmarkRoot) {
    Remove-Item -LiteralPath $benchmarkRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $benchmarkRoot | Out-Null

function Invoke-BenchmarkCase {
    param(
        [string]$Mode,
        [int]$Workers,
        [string]$Label
    )

    $outputDir = Join-Path $benchmarkRoot $Label
    if (Test-Path -LiteralPath $outputDir) {
        Remove-Item -LiteralPath $outputDir -Recurse -Force
    }

    $output = & $exe $benchmarkInput $outputDir --op edge --mode $Mode --workers $Workers 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark case $Label failed with exit code $LASTEXITCODE.`n$output"
    }

    if ($output -notmatch 'Elapsed: ([0-9]+(?:\.[0-9]+)?) ms') {
        throw "Could not parse elapsed time from $Label.`n$output"
    }
    $elapsedMs = [double]$Matches[1]

    if ($output -notmatch 'Throughput: ([0-9]+(?:\.[0-9]+)?) images/s') {
        throw "Could not parse throughput from $Label.`n$output"
    }
    $throughput = [double]$Matches[1]

    return [pscustomobject]@{
        Mode = $Mode
        Workers = $Workers
        ElapsedMs = $elapsedMs
        ImagesPerSecond = $throughput
    }
}

function Get-Median {
    param([double[]]$Values)

    $sorted = @($Values | Sort-Object)
    $middle = [math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2 -eq 1) {
        return [double]$sorted[$middle]
    }
    return ([double]$sorted[$middle - 1] + [double]$sorted[$middle]) / 2.0
}

$cases = @(
    [pscustomobject]@{ Mode = "sequential"; Workers = 1 },
    [pscustomobject]@{ Mode = "threads"; Workers = 1 },
    [pscustomobject]@{ Mode = "threads"; Workers = 2 },
    [pscustomobject]@{ Mode = "threads"; Workers = 4 }
)

$rawRows = @()
foreach ($case in $cases) {
    Invoke-BenchmarkCase -Mode $case.Mode -Workers $case.Workers -Label "$($case.Mode)-$($case.Workers)-warmup" | Out-Null

    for ($run = 1; $run -le $MeasuredRuns; $run++) {
        $result = Invoke-BenchmarkCase -Mode $case.Mode -Workers $case.Workers -Label "$($case.Mode)-$($case.Workers)-run$run"
        $rawRows += [pscustomobject]@{
            Mode = $result.Mode
            Workers = $result.Workers
            Run = $run
            ElapsedMs = $result.ElapsedMs
            ImagesPerSecond = $result.ImagesPerSecond
        }
    }
}

$summaryRows = @()
foreach ($case in $cases) {
    $matching = @($rawRows | Where-Object { $_.Mode -eq $case.Mode -and $_.Workers -eq $case.Workers })
    $summaryRows += [pscustomobject]@{
        Mode = $case.Mode
        Workers = $case.Workers
        Runs = $MeasuredRuns
        MedianElapsedMs = [math]::Round((Get-Median @($matching.ElapsedMs)), 3)
        MedianImagesPerSecond = [math]::Round((Get-Median @($matching.ImagesPerSecond)), 2)
    }
}

$baselineMs = [double]($summaryRows | Where-Object { $_.Mode -eq "sequential" }).MedianElapsedMs
foreach ($row in $summaryRows) {
    $row | Add-Member -NotePropertyName SpeedupVsSequential -NotePropertyValue ([math]::Round($baselineMs / [double]$row.MedianElapsedMs, 3))
}

$rawRows | Export-Csv -LiteralPath $rawCsv -NoTypeInformation -Encoding UTF8
$summaryRows | Export-Csv -LiteralPath $summaryCsv -NoTypeInformation -Encoding UTF8

$summaryRows | Format-Table -AutoSize | Out-String | Write-Output
Write-Output "Raw results: $rawCsv"
Write-Output "Summary: $summaryCsv"
Write-Output "DAY51_BENCHMARK_OK"
