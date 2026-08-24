$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$root = Split-Path -Parent $PSScriptRoot
$buildDir = Join-Path $root "build"
$installDir = Join-Path $buildDir "install"
$extractDir = Join-Path $buildDir "package-check"
$zip = Get-ChildItem -LiteralPath $buildDir -File -Filter "day52-cpp-image-pipeline-*.zip" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

$script:passCount = 0

function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) {
        throw "[FAIL] $Message"
    }
    $script:passCount++
    Write-Output "[PASS] $Message"
}

Assert-True ($null -ne $zip) "CPack ZIP archive exists"
Assert-True (Test-Path -LiteralPath (Join-Path $installDir "bin\day52_parallel_pipeline.exe") -PathType Leaf) "installed executable exists"
Assert-True (Test-Path -LiteralPath (Join-Path $installDir "lib\libday52_pipeline.a") -PathType Leaf) "installed static library exists"
Assert-True (Test-Path -LiteralPath (Join-Path $installDir "include\day52\image_pipeline.hpp") -PathType Leaf) "installed public header exists"
Assert-True (Test-Path -LiteralPath (Join-Path $installDir "lib\cmake\day52\day52Config.cmake") -PathType Leaf) "installed package config exists"
Assert-True (Test-Path -LiteralPath (Join-Path $installDir "lib\cmake\day52\day52ConfigVersion.cmake") -PathType Leaf) "installed version config exists"
Assert-True (Test-Path -LiteralPath (Join-Path $installDir "lib\cmake\day52\day52Targets.cmake") -PathType Leaf) "installed target export exists"

if (Test-Path -LiteralPath $extractDir) {
    Remove-Item -LiteralPath $extractDir -Recurse -Force
}
Expand-Archive -LiteralPath $zip.FullName -DestinationPath $extractDir

Assert-True (Test-Path -LiteralPath (Join-Path $extractDir "bin\day52_parallel_pipeline.exe") -PathType Leaf) "ZIP contains the executable"
Assert-True (Test-Path -LiteralPath (Join-Path $extractDir "lib\libday52_pipeline.a") -PathType Leaf) "ZIP contains the static library"
Assert-True (Test-Path -LiteralPath (Join-Path $extractDir "include\day52\image_pipeline.hpp") -PathType Leaf) "ZIP contains the public header"
Assert-True (Test-Path -LiteralPath (Join-Path $extractDir "lib\cmake\day52\day52Config.cmake") -PathType Leaf) "ZIP contains the package config"

$configText = Get-Content -LiteralPath (Join-Path $installDir "lib\cmake\day52\day52Config.cmake") -Raw
$targetsText = Get-Content -LiteralPath (Join-Path $installDir "lib\cmake\day52\day52Targets.cmake") -Raw
Assert-True ($configText -notmatch 'D:[/\\]opencv-learning') "installed config does not hard-code the source tree"
Assert-True ($targetsText -match 'day52::pipeline') "exported target is named day52::pipeline"

Write-Output "Passed $script:passCount Day52 package checks."
Write-Output "DAY52_PACKAGE_OK"
