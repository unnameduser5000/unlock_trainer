param(
    [string]$Serial = "",
    [Parameter(Mandatory = $true)] [string]$CoordinatorHost,
    [int]$CoordinatorPort = 50051,
    [Parameter(Mandatory = $true)] [string]$DeviceId,
    [int]$LocalPort = 26052,
    [switch]$Install
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$ApkPath = Join-Path $RepoRoot "app\build\outputs\apk\debug\app-debug.apk"
$AdbTarget = @()
if ($Serial -ne "") {
    $AdbTarget = @("-s", $Serial)
}

if ($Install) {
    & (Join-Path $RepoRoot "gradlew.bat") ":app:assembleDebug"
    & adb @AdbTarget install -r $ApkPath
}

& adb @AdbTarget shell am start -S `
    -n "com.example.sid_trainer/.MainActivity" `
    --es sid.coordinator_host $CoordinatorHost `
    --ei sid.coordinator_port $CoordinatorPort `
    --es sid.device_id $DeviceId `
    --ei sid.local_port $LocalPort `
    --ez sid.auto_start true
