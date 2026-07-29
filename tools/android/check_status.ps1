param(
    [string]$AdminUrl = "http://127.0.0.1:18080",
    [string[]]$Serials = @("91260221021D", "ZY22G2HC5C"),
    [string]$PackageName = "com.example.sid_trainer",
    [int]$LogLines = 120
)

$ErrorActionPreference = "Continue"

Write-Host "== ADB devices =="
adb devices -l

Write-Host ""
Write-Host "== Coordinator status =="
try {
    Invoke-WebRequest -UseBasicParsing "$AdminUrl/api/v1/status" |
        Select-Object -ExpandProperty Content
} catch {
    Write-Host "Coordinator status failed: $($_.Exception.Message)"
}

foreach ($serial in $Serials) {
    Write-Host ""
    Write-Host "== Device $serial =="
    adb -s $serial get-state

    Write-Host "-- pidof $PackageName --"
    adb -s $serial shell pidof $PackageName

    Write-Host "-- top package process --"
    adb -s $serial shell ps -A | Select-String -Pattern $PackageName

    Write-Host "-- recent focused logs --"
    adb -s $serial logcat -d -t $LogLines -s SidWorkerUi ExecuTorchShardRunner AndroidRuntime DEBUG

    Write-Host "-- tombstones --"
    adb -s $serial shell ls -lt /data/tombstones
}
