param(
    [string[]]$Serials = @("91260221021D", "ZY22G2HC5C"),
    [string]$PackageName = "com.example.sid_trainer",
    [string]$Phase = "snapshot",
    [string]$OutDir = "debug_runs"
)

$ErrorActionPreference = "Continue"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$bundle = Join-Path $OutDir "perf-$timestamp"
New-Item -ItemType Directory -Force -Path $bundle | Out-Null

adb devices -l | Out-File -Encoding utf8 (Join-Path $bundle "adb_devices.txt")

foreach ($serial in $Serials) {
    $deviceDir = Join-Path $bundle "$Phase-$serial"
    New-Item -ItemType Directory -Force -Path $deviceDir | Out-Null

    adb -s $serial get-state | Out-File -Encoding utf8 (Join-Path $deviceDir "state.txt")
    adb -s $serial shell getprop ro.product.model | Out-File -Encoding utf8 (Join-Path $deviceDir "model.txt")
    adb -s $serial shell pidof $PackageName | Out-File -Encoding utf8 (Join-Path $deviceDir "pid.txt")
    adb -s $serial shell dumpsys battery | Out-File -Encoding utf8 (Join-Path $deviceDir "battery.txt")
    adb -s $serial shell dumpsys thermalservice | Out-File -Encoding utf8 (Join-Path $deviceDir "thermalservice.txt")
    adb -s $serial shell dumpsys meminfo $PackageName | Out-File -Encoding utf8 (Join-Path $deviceDir "meminfo.txt")
    adb -s $serial shell top -b -n 1 | Out-File -Encoding utf8 (Join-Path $deviceDir "top.txt")
    adb -s $serial shell cat /sys/class/thermal/thermal_zone*/type |
        Out-File -Encoding utf8 (Join-Path $deviceDir "thermal_zone_types.txt")
    adb -s $serial shell cat /sys/class/thermal/thermal_zone*/temp |
        Out-File -Encoding utf8 (Join-Path $deviceDir "thermal_zone_temps.txt")
    adb -s $serial logcat -d -t 800 -s SidWorkerUi ExecuTorchShardRunner GrpcManager AndroidRuntime DEBUG |
        Out-File -Encoding utf8 (Join-Path $deviceDir "focused_logcat.txt")
}

Write-Host "Perf snapshot written to $bundle"
