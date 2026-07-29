param(
    [string[]]$Serials = @("91260221021D", "ZY22G2HC5C"),
    [string]$PackageName = "com.example.sid_trainer",
    [string]$Phase = "run",
    [int]$IntervalSec = 2,
    [int]$MeminfoEverySamples = 5,
    [int]$DurationSec = 0,
    [string]$OutDir = "debug_runs"
)

$ErrorActionPreference = "Continue"

if ($IntervalSec -lt 1) {
    throw "IntervalSec must be >= 1."
}
if ($DurationSec -lt 0) {
    throw "DurationSec must be >= 0. Use 0 to run until Ctrl+C."
}
if ($MeminfoEverySamples -lt 1) {
    throw "MeminfoEverySamples must be >= 1."
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$bundle = Join-Path $OutDir "perf-monitor-$timestamp"
New-Item -ItemType Directory -Force -Path $bundle | Out-Null

$samplesCsv = Join-Path $bundle "samples.csv"
$eventsCsv = Join-Path $bundle "events.csv"

$sampleHeader = @(
    "timestamp_iso",
    "elapsed_ms",
    "phase",
    "serial",
    "model",
    "pid",
    "battery_level",
    "battery_status",
    "battery_temp_c",
    "battery_voltage_mv",
    "battery_current_ua",
    "thermal_status",
    "app_pss_kb",
    "app_private_dirty_kb",
    "proc_rss_kb",
    "proc_vsz_kb"
) -join ","

$eventHeader = @(
    "timestamp_iso",
    "elapsed_ms",
    "phase",
    "serial",
    "tag",
    "message"
) -join ","

Set-Content -Encoding utf8 -Path $samplesCsv -Value $sampleHeader
Set-Content -Encoding utf8 -Path $eventsCsv -Value $eventHeader
adb devices -l | Out-File -Encoding utf8 (Join-Path $bundle "adb_devices.txt")
$seenLogEvents = @{}

function CsvEscape([object]$Value) {
    if ($null -eq $Value) {
        return ""
    }
    $text = [string]$Value
    $escaped = $text.Replace('"', '""')
    if ($escaped.Contains(",") -or $escaped.Contains('"') -or $escaped.Contains("`n") -or $escaped.Contains("`r")) {
        return '"' + $escaped + '"'
    }
    return $escaped
}

function To-CsvRow([object[]]$Values) {
    return ($Values | ForEach-Object { CsvEscape $_ }) -join ","
}

function Get-AdbShellLines([string]$Serial, [string]$Command) {
    $output = adb -s $Serial shell $Command 2>$null
    if ($null -eq $output) {
        return @()
    }
    return @($output)
}

function Get-AdbShellText([string]$Serial, [string]$Command) {
    return (Get-AdbShellLines -Serial $Serial -Command $Command) -join "`n"
}

function Parse-KeyValueLines([string[]]$Lines) {
    $values = @{}
    foreach ($line in $Lines) {
        if ($line -match "^\s*([^:]+):\s*(.*)$") {
            $values[$matches[1].Trim()] = $matches[2].Trim()
        }
    }
    return $values
}

function Get-BatteryCurrentUa([string]$Serial) {
    $value = Get-AdbShellText -Serial $Serial -Command "cat /sys/class/power_supply/battery/current_now"
    if ($value -match "^-?\d+") {
        return $matches[0]
    }
    return ""
}

function Get-ThermalStatus([string]$Serial) {
    $thermal = Get-AdbShellLines -Serial $Serial -Command "dumpsys thermalservice"
    foreach ($line in $thermal) {
        if ($line -match "Thermal Status:\s*(\S+)") {
            return $matches[1]
        }
        if ($line -match "^\s*status:\s*(\S+)") {
            return $matches[1]
        }
    }
    return ""
}

function Get-AppMeminfo([string]$Serial, [string]$PackageName) {
    $meminfo = Get-AdbShellLines -Serial $Serial -Command "dumpsys meminfo $PackageName"
    $result = @{
        PssKb = ""
        PrivateDirtyKb = ""
    }
    foreach ($line in $meminfo) {
        if ($line -match "TOTAL PSS:\s*(\d+)") {
            $result.PssKb = $matches[1]
        }
        if ($line -match "TOTAL\s+(\d+)\s+(\d+)") {
            if ([string]::IsNullOrWhiteSpace($result.PssKb)) {
                $result.PssKb = $matches[1]
            }
            if ([string]::IsNullOrWhiteSpace($result.PrivateDirtyKb)) {
                $result.PrivateDirtyKb = $matches[2]
            }
        }
    }
    return $result
}

function Get-ProcessStats([string]$Serial, [string]$PackageName) {
    $psLines = Get-AdbShellLines -Serial $Serial -Command "ps -A"
    $result = @{
        Pid = ""
        RssKb = ""
        VszKb = ""
    }
    $line = $psLines | Where-Object { $_ -match [regex]::Escape($PackageName) } | Select-Object -First 1
    if ($line) {
        $columns = ($line.Trim() -split "\s+")
        if ($columns.Count -ge 9) {
            $result.Pid = $columns[1]
            $result.VszKb = $columns[3]
            $result.RssKb = $columns[4]
        } elseif ($columns.Count -ge 2) {
            $result.Pid = $columns[1]
        }
    }
    if ([string]::IsNullOrWhiteSpace($result.Pid)) {
        $pidText = Get-AdbShellText -Serial $Serial -Command "pidof $PackageName"
        if ($pidText -match "\d+") {
            $result.Pid = $matches[0]
        }
    }
    return $result
}

function Append-RecentTimingEvents(
    [string]$Serial,
    [string]$Phase,
    [long]$ElapsedMs,
    [string]$EventsCsv
) {
    $logLines = Get-AdbShellLines -Serial $Serial -Command "logcat -d -t 120 -s SidWorkerUi ExecuTorchShardRunner GrpcManager"
    foreach ($line in $logLines) {
        if (
            $line -match "Local shard finished" -or
            $line -match "Shard execution timing" -or
            $line -match "Forward completed" -or
            $line -match "executeForwardBackward\(\).*SGD.step"
        ) {
            $eventKey = "$Serial|$line"
            if ($seenLogEvents.ContainsKey($eventKey)) {
                continue
            }
            $seenLogEvents[$eventKey] = $true
            $row = To-CsvRow @(
                (Get-Date).ToString("o"),
                $ElapsedMs,
                $Phase,
                $Serial,
                "logcat",
                $line
            )
            Add-Content -Encoding utf8 -Path $EventsCsv -Value $row
        }
    }
}

$startedAt = Get-Date
$stopAt = if ($DurationSec -gt 0) { $startedAt.AddSeconds($DurationSec) } else { $null }
$sampleIndex = 0

Write-Host "Writing samples to $samplesCsv"
Write-Host "Writing timing events to $eventsCsv"
Write-Host "Press Ctrl+C to stop when DurationSec=0."

while ($true) {
    $now = Get-Date
    if ($null -ne $stopAt -and $now -ge $stopAt) {
        break
    }
    $elapsedMs = [int64](($now - $startedAt).TotalMilliseconds)

    foreach ($serial in $Serials) {
        $model = (Get-AdbShellText -Serial $serial -Command "getprop ro.product.model").Trim()
        $batteryLines = Get-AdbShellLines -Serial $serial -Command "dumpsys battery"
        $battery = Parse-KeyValueLines $batteryLines
        $mem = if (($sampleIndex % $MeminfoEverySamples) -eq 0) {
            Get-AppMeminfo -Serial $serial -PackageName $PackageName
        } else {
            @{ PssKb = ""; PrivateDirtyKb = "" }
        }
        $proc = Get-ProcessStats -Serial $serial -PackageName $PackageName
        $batteryTempC = ""
        if ($battery.ContainsKey("temperature") -and $battery["temperature"] -match "^-?\d+$") {
            $batteryTempC = ([double]$battery["temperature"] / 10.0).ToString("0.0")
        }

        $row = To-CsvRow @(
            $now.ToString("o"),
            $elapsedMs,
            $Phase,
            $serial,
            $model,
            $proc["Pid"],
            $battery["level"],
            $battery["status"],
            $batteryTempC,
            $battery["voltage"],
            (Get-BatteryCurrentUa -Serial $serial),
            (Get-ThermalStatus -Serial $serial),
            $mem["PssKb"],
            $mem["PrivateDirtyKb"],
            $proc["RssKb"],
            $proc["VszKb"]
        )
        Add-Content -Encoding utf8 -Path $samplesCsv -Value $row

        if ($sampleIndex -eq 0 -or ($sampleIndex % 5) -eq 0) {
            Append-RecentTimingEvents -Serial $serial -Phase $Phase -ElapsedMs $elapsedMs -EventsCsv $eventsCsv
        }
    }

    $sampleIndex++
    Start-Sleep -Seconds $IntervalSec
}

foreach ($serial in $Serials) {
    adb -s $serial logcat -d -t 1000 -s SidWorkerUi ExecuTorchShardRunner GrpcManager AndroidRuntime DEBUG |
        Out-File -Encoding utf8 (Join-Path $bundle "focused_logcat_$serial.txt")
}

Write-Host "Perf monitor written to $bundle"
