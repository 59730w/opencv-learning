param(
    [Parameter(Mandatory = $true)]
    [string]$Executable,

    [Parameter(Mandatory = $true)]
    [string]$Model
)

$ErrorActionPreference = 'Stop'

function Assert-Contains {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Text,

        [Parameter(Mandatory = $true)]
        [string]$Expected
    )

    if (-not $Text.Contains($Expected)) {
        throw "Expected output to contain: $Expected`nActual output:`n$Text"
    }
}

if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "Executable does not exist: $Executable"
}

if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw "Model does not exist: $Model"
}

$outputLines = & $Executable $Model 2>&1
$exitCode = $LASTEXITCODE
$output = $outputLines -join [Environment]::NewLine

if ($exitCode -ne 0) {
    throw "Expected exit code 0, got $exitCode.`nOutput:`n$output"
}

Assert-Contains -Text $output -Expected 'Input count: 1'
Assert-Contains -Text $output -Expected 'Input 0 name: images'
Assert-Contains -Text $output -Expected 'Input 0 type: float32'
Assert-Contains -Text $output -Expected 'Input 0 shape: [batch, 3, 224, 224]'
Assert-Contains -Text $output -Expected 'Output count: 1'
Assert-Contains -Text $output -Expected 'Output 0 name: logits'
Assert-Contains -Text $output -Expected 'Output 0 type: float32'
Assert-Contains -Text $output -Expected 'Output 0 shape: [batch, 50]'
Assert-Contains -Text $output -Expected 'DAY54_MODEL_INTERFACE_OK'

Write-Output 'DAY54_CONTRACT_OK'
