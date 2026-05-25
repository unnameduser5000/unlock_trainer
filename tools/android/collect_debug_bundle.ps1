param(
    [string]$AdminUrl = "http://127.0.0.1:18080",
    [string[]]$Serials = @("91260221021D", "ZY22G2HC5C"),
    [string]$PackageName = "com.example.sid_trainer",
    [string]$OutDir = "debug_runs"
)

$ErrorActionPreference = "Continue"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$bundle = Join-Path $OutDir "android-$timestamp"
New-Item -ItemType Directory -Force -Path $bundle | Out-Null

adb devices -l | Out-File -Encoding utf8 (Join-Path $bundle "adb_devices.txt")

try {
    Invoke-WebRequest -UseBasicParsing "$AdminUrl/api/v1/status" |
        Select-Object -ExpandProperty Content |
        Out-File -Encoding utf8 (Join-Path $bundle "coordinator_status.json")
} catch {
    "Coordinator status failed: $($_.Exception.Message)" |
        Out-File -Encoding utf8 (Join-Path $bundle "coordinator_status_error.txt")
}

try {
    Invoke-WebRequest -UseBasicParsing "$AdminUrl/api/v1/requests" |
        Select-Object -ExpandProperty Content |
        Out-File -Encoding utf8 (Join-Path $bundle "coordinator_requests.json")
} catch {
    "Coordinator requests failed: $($_.Exception.Message)" |
        Out-File -Encoding utf8 (Join-Path $bundle "coordinator_requests_error.txt")
}

foreach ($serial in $Serials) {
    $deviceDir = Join-Path $bundle $serial
    New-Item -ItemType Directory -Force -Path $deviceDir | Out-Null

    adb -s $serial get-state | Out-File -Encoding utf8 (Join-Path $deviceDir "state.txt")
    adb -s $serial shell pidof $PackageName | Out-File -Encoding utf8 (Join-Path $deviceDir "pid.txt")
    adb -s $serial shell ps -A | Out-File -Encoding utf8 (Join-Path $deviceDir "ps.txt")
    adb -s $serial shell dumpsys meminfo $PackageName | Out-File -Encoding utf8 (Join-Path $deviceDir "meminfo.txt")
    adb -s $serial logcat -d -t 1000 -s SidWorkerUi ExecuTorchShardRunner AndroidRuntime DEBUG |
        Out-File -Encoding utf8 (Join-Path $deviceDir "focused_logcat.txt")
    adb -s $serial shell ls -lt /data/tombstones |
        Out-File -Encoding utf8 (Join-Path $deviceDir "tombstones_list.txt")
}

Write-Host "Debug bundle written to $bundle"
