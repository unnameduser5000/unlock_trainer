param(
    [string]$AdminUrl = "http://127.0.0.1:18080",
    [string]$PipelineConfig = "coordinator/config/pipeline.json",
    [string[]]$Serials = @("91260221021D", "ZY22G2HC5C"),
    [string]$CoordinatorHost = "",
    [int]$CoordinatorGrpcPort = 50051,
    [int]$CoordinatorHttpPort = 18080,
    [int]$WorkerPort = 26052,
    [int]$TimeoutMs = 2500
)

$ErrorActionPreference = "Continue"

function Write-Section([string]$Title) {
    Write-Host ""
    Write-Host "== $Title =="
}

function Test-TcpPort([string]$HostName, [int]$Port, [int]$TimeoutMs) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect($HostName, $Port, $null, $null)
        $ok = $async.AsyncWaitHandle.WaitOne($TimeoutMs)
        if (-not $ok) {
            return $false
        }
        $client.EndConnect($async)
        return $client.Connected
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

function Get-AdbShellText([string]$Serial, [string]$Command) {
    $output = adb -s $Serial shell $Command 2>$null
    if ($null -eq $output) {
        return ""
    }
    return (@($output) -join "`n").Trim()
}

function Get-DeviceIpv4([string]$Serial) {
    $ipText = Get-AdbShellText -Serial $Serial -Command "ip -4 addr show wlan0"
    if ($ipText -match "inet\s+([0-9.]+)/") {
        return $matches[1]
    }
    return ""
}

function Test-DeviceTcpPort([string]$Serial, [string]$HostName, [int]$Port, [int]$TimeoutMs) {
    $timeoutSec = [Math]::Max(1, [Math]::Ceiling($TimeoutMs / 1000.0))
    $cmd = "toybox nc -z -w $timeoutSec $HostName $Port >/dev/null 2>&1; echo PROBE_STATUS:`$?"
    $output = Get-AdbShellText -Serial $Serial -Command $cmd
    if ($output -match "PROBE_STATUS:0") {
        return "open"
    }
    if ($output -match "not found|No such file|unknown") {
        return "unsupported"
    }
    return "closed"
}

function Get-JsonStatus([string]$Url) {
    try {
        return (Invoke-WebRequest -UseBasicParsing $Url -TimeoutSec 5 | Select-Object -ExpandProperty Content) |
            ConvertFrom-Json
    } catch {
        Write-Host "Coordinator admin query failed: $($_.Exception.Message)"
        return $null
    }
}

function Get-HostFromUrl([string]$UrlText) {
    try {
        return ([Uri]$UrlText).Host
    } catch {
        return ""
    }
}

Write-Section "PC IPv4 addresses"
ipconfig | Select-String -Pattern "IPv4|Wireless|Wi-Fi|Ethernet|adapter|Adapter" | ForEach-Object {
    Write-Host $_.Line
}

if ([string]::IsNullOrWhiteSpace($CoordinatorHost) -and (Test-Path $PipelineConfig)) {
    try {
        $config = Get-Content $PipelineConfig -Raw | ConvertFrom-Json
        $CoordinatorHost = Get-HostFromUrl $config.artifactBaseUrl
    } catch {
        Write-Host "Could not read ${PipelineConfig}: $($_.Exception.Message)"
    }
}

Write-Section "Coordinator"
Write-Host "CoordinatorHost=$CoordinatorHost"
Write-Host "CoordinatorGrpcPort=$CoordinatorGrpcPort"
Write-Host "CoordinatorHttpPort=$CoordinatorHttpPort"
Write-Host "AdminUrl=$AdminUrl"
$status = Get-JsonStatus "$AdminUrl/api/v1/status"
if ($null -ne $status) {
    Write-Host "summary live=$($status.summary.liveNodeCount) inactive=$($status.summary.inactiveNodeCount) offline=$($status.summary.offlineStageCount) epoch=$($status.summary.routingEpoch)"
    foreach ($stage in $status.stages) {
        $node = $stage.assignedNode
        Write-Host "stage=$($stage.stageId) ready=$($stage.routeReady) expected=$($stage.expectedHost):$($stage.expectedPort) node=$($node.nodeId) ip=$($node.ipAddress):$($node.grpcPort) live=$($node.isLive)"
    }
}

Write-Section "ADB devices"
adb devices -l

$deviceInfos = @()
foreach ($serial in $Serials) {
    $model = Get-AdbShellText -Serial $serial -Command "getprop ro.product.model"
    $ip = Get-DeviceIpv4 -Serial $serial
    $route = Get-AdbShellText -Serial $serial -Command "ip route"
    $pidText = Get-AdbShellText -Serial $serial -Command "pidof com.example.sid_trainer"
    $deviceInfos += [pscustomobject]@{
        Serial = $serial
        Model = $model
        Ip = $ip
        Pid = $pidText
        Route = $route
    }
}

Write-Section "Device addresses"
$deviceInfos | Format-Table Serial,Model,Ip,Pid -AutoSize
foreach ($info in $deviceInfos) {
    Write-Host "-- route $($info.Serial) --"
    Write-Host $info.Route
}

Write-Section "PC to worker ports"
foreach ($info in $deviceInfos) {
    if ([string]::IsNullOrWhiteSpace($info.Ip)) {
        Write-Host "$($info.Serial) ip=missing result=skip"
        continue
    }
    $open = Test-TcpPort -HostName $info.Ip -Port $WorkerPort -TimeoutMs $TimeoutMs
    Write-Host "pc -> $($info.Serial) $($info.Ip):$WorkerPort $open"
}

Write-Section "Device to coordinator ports"
foreach ($info in $deviceInfos) {
    if ([string]::IsNullOrWhiteSpace($CoordinatorHost)) {
        Write-Host "$($info.Serial) coordinatorHost=missing result=skip"
        continue
    }
    $grpc = Test-DeviceTcpPort -Serial $info.Serial -HostName $CoordinatorHost -Port $CoordinatorGrpcPort -TimeoutMs $TimeoutMs
    $http = Test-DeviceTcpPort -Serial $info.Serial -HostName $CoordinatorHost -Port $CoordinatorHttpPort -TimeoutMs $TimeoutMs
    Write-Host "$($info.Serial) -> coordinator $CoordinatorHost grpc:$CoordinatorGrpcPort=$grpc http:$CoordinatorHttpPort=$http"
}

Write-Section "Device to device worker ports"
foreach ($src in $deviceInfos) {
    foreach ($dst in $deviceInfos) {
        if ($src.Serial -eq $dst.Serial) {
            continue
        }
        if ([string]::IsNullOrWhiteSpace($dst.Ip)) {
            Write-Host "$($src.Serial) -> $($dst.Serial) ip=missing result=skip"
            continue
        }
        $result = Test-DeviceTcpPort -Serial $src.Serial -HostName $dst.Ip -Port $WorkerPort -TimeoutMs $TimeoutMs
        Write-Host "$($src.Serial) -> $($dst.Serial) $($dst.Ip):$WorkerPort $result"
    }
}

Write-Section "Interpretation"
Write-Host "For this pipeline, all three directions should be open:"
Write-Host "1. device -> coordinator $CoordinatorGrpcPort and $CoordinatorHttpPort"
Write-Host "2. PC/coordinator -> each device $WorkerPort"
Write-Host "3. stage device -> next-stage device $WorkerPort"
Write-Host "If Windows Mobile Hotspot isolates clients, check #3 will fail even when phones can reach the PC."
