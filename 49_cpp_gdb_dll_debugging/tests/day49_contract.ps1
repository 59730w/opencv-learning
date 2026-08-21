$ErrorActionPreference = 'Stop'

$day49Root = Split-Path -Parent $PSScriptRoot
$debugExe = Join-Path $day49Root 'build\day49_debug_target.exe'
$inputImage = Join-Path $day49Root 'assets\day45_input.jpg'
$breakpointScript = Join-Path $PSScriptRoot 'day49_breakpoint.gdb'
$exceptionScript = Join-Path $PSScriptRoot 'day49_exception.gdb'
$sharedLibrariesScript = Join-Path $PSScriptRoot 'day49_shared_libraries.gdb'
$failures = 0

function Assert-Day49 {
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

Assert-Day49 (Test-Path -LiteralPath $debugExe -PathType Leaf) `
    'debug executable exists'

if ($failures -gt 0) {
    Write-Host "DAY49_CONTRACT_FAILED failures=$failures"
    exit 1
}

if (-not $env:MSYS2_ROOT) {
    throw 'MSYS2_ROOT must point to the MSYS2 installation root'
}

$gdb = Join-Path $env:MSYS2_ROOT 'ucrt64\bin\gdb.exe'
$objdump = Join-Path $env:MSYS2_ROOT 'ucrt64\bin\objdump.exe'
Assert-Day49 (Test-Path -LiteralPath $gdb -PathType Leaf) 'gdb is available'
Assert-Day49 (Test-Path -LiteralPath $objdump -PathType Leaf) 'objdump is available'

$validOutput = (& $debugExe $inputImage 1 2>&1 | Out-String)
$validExit = $LASTEXITCODE
Assert-Day49 ($validExit -eq 0) 'valid summary index exits with 0'
Assert-Day49 ($validOutput -match 'Image: 427 x 640 channels=3') `
    'valid run reports image dimensions and channels'
Assert-Day49 ($validOutput -match 'summary\[1\]=640') `
    'valid run selects the image height'
Assert-Day49 ($validOutput -match 'DAY49_DEBUG_OK') `
    'valid run prints the success marker'

$null = & $debugExe (Join-Path $day49Root 'assets\missing.jpg') 1 2>&1
Assert-Day49 ($LASTEXITCODE -eq 1) 'missing image exits with 1'

$null = & $debugExe 2>&1
Assert-Day49 ($LASTEXITCODE -eq 2) 'missing arguments exit with 2'

$null = & $debugExe $inputImage 'abc' 2>&1
Assert-Day49 ($LASTEXITCODE -eq 3) 'non-integer index exits with 3'

$rangeOutput = (& $debugExe $inputImage 9 2>&1 | Out-String)
$rangeExit = $LASTEXITCODE
Assert-Day49 ($rangeExit -eq 4) 'out-of-range index exits with 4'
Assert-Day49 ($rangeOutput -match 'Summary index out of range: 9') `
    'out-of-range index has a deterministic error message'

$breakpointOutput = (& $gdb -q -batch -x $breakpointScript $debugExe 2>&1 | Out-String)
Assert-Day49 ($breakpointOutput -match 'Breakpoint 1, main') `
    'GDB stops at main with source symbols'
Assert-Day49 ($breakpointOutput -match 'Breakpoint 2, select_summary_value') `
    'GDB stops in the selected helper function'
Assert-Day49 ($breakpointOutput -match 'index = 1') `
    'GDB prints the valid index argument'
Assert-Day49 ($breakpointOutput -match 'day49_gdb_dll_debugging\.cpp') `
    'GDB output contains the Day49 source filename'

$exceptionOutput = (& $gdb -q -batch -x $exceptionScript $debugExe 2>&1 | Out-String)
Assert-Day49 ($exceptionOutput -match 'hit Breakpoint 1') `
    'GDB stops at the MinGW out-of-range throw helper'
Assert-Day49 ($exceptionOutput -match 'select_summary_value') `
    'exception backtrace reaches Day49 code'
Assert-Day49 ($exceptionOutput -match 'index = 9') `
    'exception investigation exposes the invalid index'

$dllOutput = (& $objdump -p $debugExe 2>&1 | Out-String)
Assert-Day49 ($dllOutput -match 'DLL Name: libopencv_core-500\.dll') `
    'PE imports include the OpenCV core DLL'
Assert-Day49 ($dllOutput -match 'DLL Name: libopencv_imgcodecs-500\.dll') `
    'PE imports include the OpenCV imgcodecs DLL'
Assert-Day49 ($dllOutput -match 'DLL Name: libstdc\+\+-6\.dll') `
    'PE imports include the MinGW C++ runtime DLL'

$sharedOutput = (& $gdb -q -batch -x $sharedLibrariesScript $debugExe 2>&1 | Out-String)
Assert-Day49 ($sharedOutput -match 'Shared Object Library') `
    'GDB reports loaded shared libraries'
Assert-Day49 ($sharedOutput -match 'opencv_core-500') `
    'GDB sees the loaded OpenCV core DLL'

if ($failures -gt 0) {
    Write-Host "DAY49_CONTRACT_FAILED failures=$failures"
    exit 1
}

Write-Host 'DAY49_CONTRACT_OK'
exit 0
