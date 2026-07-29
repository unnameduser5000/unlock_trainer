param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,

    [string]$AdminBaseUrl = "http://127.0.0.1:18080",
    [string]$OutDir = "",
    [int]$Limit = 100000,
    [switch]$GenerateFigures
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($OutDir)) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutDir = Join-Path "debug_runs" "coordinator-export-$RunId-$timestamp"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$encodedRunId = [System.Uri]::EscapeDataString($RunId)
$base = $AdminBaseUrl.TrimEnd("/")
$metricsPath = Join-Path $OutDir "metrics.csv"
$stageTimingsPath = Join-Path $OutDir "stage-timings.csv"
$summaryPath = Join-Path $OutDir "run-summary.json"

Invoke-WebRequest -UseBasicParsing "$base/api/v1/runs/$encodedRunId`?metrics=1000" -OutFile $summaryPath
Invoke-WebRequest -UseBasicParsing "$base/api/v1/runs/$encodedRunId/metrics.csv`?limit=$Limit" -OutFile $metricsPath
Invoke-WebRequest -UseBasicParsing "$base/api/v1/runs/$encodedRunId/stage-timings.csv`?limit=$Limit" -OutFile $stageTimingsPath

Write-Host "Wrote coordinator export:"
Write-Host "  $summaryPath"
Write-Host "  $metricsPath"
Write-Host "  $stageTimingsPath"

if ($GenerateFigures) {
    $figuresDir = Join-Path $OutDir "figures"
    python tools\report\generate_demo_figures.py --coordinator_run_dir $OutDir --output_dir $figuresDir
    Write-Host "Wrote figures:"
    Write-Host "  $figuresDir"
}
